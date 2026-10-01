"""
isobank.training
================

Training driver for the isometric flow.

Assembles the pair sets (long-range / local / boundary and an optional
dedicated EDGE set), a held-out monitoring set, and runs the validated loss
recipe

	loss = w_g * L_global + w_l * L_local + w_b * L_bound [+ w_e * L_edge]

with Adam + cosine LR annealing, gradient clipping, and checkpoint selection
on p10 (10th percentile of d_z/d_proper at placement scale).

The EDGE set covers a known-hard sub-region (e.g. high eccentricity):
boundary-scale pairs whose anchors are drawn uniformly in that sub-region.
Without it, the tiered data mix alone under-weights the edge in the
placement-scale losses. When an edge set is given, the checkpoint criterion
becomes  max min(p10_monitor, p10_edge)  so the kept epoch is good on BOTH
the bulk and the edge.
"""

import sys
import numpy as np
import torch
from torch import optim

from .flow import IsometricFlow, save_checkpoint
from .losses import (ratio_loss_with_tail, log_distance_loss,
                     scale_percentiles, global_correlation)

__all__ = ['train_isometric_flow']

####################################################################################################################

def train_isometric_flow(metric, r_target, variable_format,
                         long_pairs, local_pairs, bound_pairs, monitor_pairs,
                         d_flow, n_layers, hidden_dim=256, masks=None, depth=3,
                         edge_pairs=None, degen_pairs=None, w_degen=10.0,
                         weights=(1.0, 5.0, 10.0, 10.0),
                         tail_frac=0.25, tail_w=5.0,
                         embed_fn=None, train_data=None, phi_col=None,
                         n_epochs=1000, steps_per_epoch=79, lr=5e-4,
                         best_path='flow_best.pt', final_path='flow_final.pt',
                         extra_ckpt=None, verbose_every=20):
	"""
	Train an isometric flow.

	Parameters
	----------
		long_pairs, local_pairs, bound_pairs, monitor_pairs: tuple
			Each a tuple ``(t1, t2, d_true)`` of theta-space endpoints and
			proper distances (see :mod:`isobank.pairs`). ``monitor_pairs``
			must come from held-out anchors.
		edge_pairs: tuple
			Optional ``(t1, t2, d_true)`` boundary-scale pairs anchored in a
			known-hard sub-region; adds the L_edge loss term and switches the
			checkpoint criterion to max min(p10_monitor, p10_edge).
		degen_pairs: tuple
			Optional ``(t1, t2, d_true)`` pairs stepping along the metric's
			near-degenerate eigenvectors (see
			:func:`isobank.pairs.degenerate_direction_pairs`). Adds
			``w_degen * L_degen`` (ratio loss) which pulls the flow to map the
			soft directions to their small true proper distance, removing the
			template redundancy that inflates the bank. The soft-direction
			ratio is monitored (``degen_med``) so over-expansion (ratio > 1)
			can be seen shrinking toward 1 during training.
		weights: tuple
			(w_global, w_local, w_bound, w_edge)
		depth: int
			Hidden-layer count of the coupling conditioners
		tail_frac, tail_w: float
			Tail-penalty parameters of the ratio losses
		embed_fn: callable
			Maps theta tensors to flow inputs (e.g. sphere lift); identity
			if None. Applied once to all pair sets before training.

	Returns
	-------
		flow: :class:`isobank.flow.IsometricFlow`
			The final flow (NOT necessarily the best — reload ``best_path``)
		history: dict
			epoch losses and monitor curves
	"""
	if embed_fn is None: embed_fn = lambda t: t
	extra_ckpt = extra_ckpt or {}
	w_g, w_l, w_b, w_e = weights

		#lift all pair sets once
	long_x1, long_x2, long_d = embed_fn(long_pairs[0]),  embed_fn(long_pairs[1]),  long_pairs[2]
	lx1, lx2, ld_true        = embed_fn(local_pairs[0]), embed_fn(local_pairs[1]), local_pairs[2]
	bx1, bx2, bd_true        = embed_fn(bound_pairs[0]), embed_fn(bound_pairs[1]), bound_pairs[2]
	mx1, mx2, md_true        = embed_fn(monitor_pairs[0]), embed_fn(monitor_pairs[1]), monitor_pairs[2]
	if edge_pairs is not None:
		ex1, ex2, ed_true = embed_fn(edge_pairs[0]), embed_fn(edge_pairs[1]), edge_pairs[2]
	if degen_pairs is not None:
		gx1, gx2, gd_true = embed_fn(degen_pairs[0]), embed_fn(degen_pairs[1]), degen_pairs[2]

	flow      = IsometricFlow(d=d_flow, n_layers=n_layers, hidden_dim=hidden_dim,
	                          masks=masks, depth=depth)
	optimizer = optim.Adam(flow.parameters(), lr=lr)
	scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-6)

	n_params = sum(p.numel() for p in flow.parameters())
	print(f"Training IsometricFlow | d={d_flow} | {n_layers} layers (depth={depth}) | "
	      f"{n_params:,} params | {n_epochs} epochs")
	print(f"Pairs: long={len(long_d)}  local={len(ld_true)}  "
	      f"boundary={len(bd_true)}  monitor={len(md_true)}"
	      + (f"  edge={len(ed_true)}" if edge_pairs is not None else ""))
	print(f"Loss weights: L_global x{w_g}  L_local x{w_l}  L_bound x{w_b}"
	      + (f"  L_edge x{w_e}" if edge_pairs is not None else ""))
	print(f"Tail penalty: top-{int(tail_frac*100)}% pairs weighted {1+tail_w:.0f}x")
	print("Checkpoint criterion: " +
	      ("max min(p10_monitor, p10_edge)" if edge_pairs is not None
	       else "p10_monitor closest to 1"))
	sys.stdout.flush()

	def L_global(n=128):
		idx = torch.randint(0, len(long_d), (n,))
		z1, _ = flow(long_x1[idx]); z2, _ = flow(long_x2[idx])
		return log_distance_loss(torch.norm(z1 - z2, dim=1), long_d[idx])

	def L_pairs(x1, x2, d_true, n=256):
		n_use = min(n, len(d_true))
		idx   = torch.randint(0, len(d_true), (n_use,))
		z1, _ = flow(x1[idx]); z2, _ = flow(x2[idx])
		return ratio_loss_with_tail(torch.norm(z1 - z2, dim=1), d_true[idx],
		                            tail_frac=tail_frac, tail_w=tail_w)

	best_score = -float('inf')
	history = dict(loss=[], monitor_med=[], monitor_p10=[], edge_p10=[], epochs=[])

	from tqdm import trange
	for epoch in trange(n_epochs):
		flow.train()
		epoch_loss = 0.0
		for _ in range(steps_per_epoch):
			optimizer.zero_grad()
			lg = L_global(n=128)
			ll = L_pairs(lx1, lx2, ld_true, n=256)
			lb = L_pairs(bx1, bx2, bd_true, n=256)
			loss = w_g * lg + w_l * ll + w_b * lb
			if edge_pairs is not None:
				le = L_pairs(ex1, ex2, ed_true, n=256)
				loss = loss + w_e * le
			if degen_pairs is not None:
				lgd = L_pairs(gx1, gx2, gd_true, n=256)
				loss = loss + w_degen * lgd
			loss.backward()
			torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
			optimizer.step()
			epoch_loss += loss.item()
		scheduler.step()
		history['loss'].append(epoch_loss / steps_per_epoch)

		if epoch % verbose_every == 0:
			flow.eval()
			tr_med, _, tr_p10 = scale_percentiles(flow, bx1, bx2, bd_true)
			mo_med, mo_p25, mo_p10 = scale_percentiles(flow, mx1, mx2, md_true)
			if edge_pairs is not None:
				e_med, _, e_p10 = scale_percentiles(flow, ex1, ex2, ed_true)
			if degen_pairs is not None:
				gd_med, _, gd_p10 = scale_percentiles(flow, gx1, gx2, gd_true)
			g_corr = (global_correlation(flow, metric, train_data,
			                             phi_col=phi_col, embed_fn=embed_fn)
			          if train_data is not None else float('nan'))
			history['epochs'].append(epoch)
			history['monitor_med'].append(mo_med)
			history['monitor_p10'].append(mo_p10)
			history['edge_p10'].append(e_p10 if edge_pairs is not None else float('nan'))
			msg = (f"Epoch {epoch:03d} | loss={history['loss'][-1]:.4f} | "
			       f"L_g={lg.item():.4f} L_l={ll.item():.4f} L_b={lb.item():.4f}"
			       + (f" L_e={le.item():.4f}" if edge_pairs is not None else "")
			       + (f" L_d={lgd.item():.4f}" if degen_pairs is not None else "") +
			       f" | train  med={tr_med:.4f} p10={tr_p10:.4f} | "
			       f"monitor med={mo_med:.4f} p10={mo_p10:.4f}"
			       + (f" | edge med={e_med:.4f} p10={e_p10:.4f}" if edge_pairs is not None else "")
			       + (f" | degen med={gd_med:.4f} (want ->1)" if degen_pairs is not None else "") +
			       f" | global_r={g_corr:.3f}")
			print(msg)

			score = min(mo_p10, e_p10) if edge_pairs is not None else -abs(mo_p10 - 1.0)
			if score > best_score:
				best_score = score
				extra = dict(extra_ckpt)
				if edge_pairs is not None:
					extra['scale_p10_edge'] = e_p10
				save_checkpoint(best_path, flow, r_target, variable_format,
				                epoch=epoch, scale_median=mo_med,
				                scale_p25=mo_p25, scale_p10=mo_p10,
				                scale_median_train=tr_med, scale_p10_train=tr_p10,
				                global_r=g_corr, **extra)
				print(f"  *** Best (monitor p10={mo_p10:.4f}  train p10={tr_p10:.4f}"
				      + (f"  edge p10={e_p10:.4f}" if edge_pairs is not None else "")
				      + f"  gap={tr_p10-mo_p10:+.4f})")
			sys.stdout.flush()

	save_checkpoint(final_path, flow, r_target, variable_format, **extra_ckpt)
	print(f"\nSaved final: {final_path}")
	print(f"Saved best:  {best_path}")
	return flow, history
