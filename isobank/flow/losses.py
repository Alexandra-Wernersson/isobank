"""
isobank.losses
==============

Loss functions and monitoring utilities for isometric-flow training.

The training recipe (validated across 2D/4D/5D spaces):

	loss = 1.0 * L_global + 5.0 * L_local + 10.0 * L_bound

- ``L_global`` — squared log-ratio on long-range pairs: keeps the global
  layout of the space consistent without letting far pairs dominate.
- ``L_local`` — ratio loss on metric-guided local pairs.
- ``L_bound`` — ratio loss on boundary pairs at [0.5, 1.5] x r_target, the
  scale template placement actually operates at; hence its large weight.


"""

import numpy as np
import torch

__all__ = ['ratio_loss_with_tail', 'log_distance_loss', 'scale_percentiles',
           'global_correlation']

####################################################################################################################

def ratio_loss_with_tail(d_z, d_proper, tail_frac=0.25, tail_w=5.0, clamp=5.0):
	"""
	Mean squared error of the ratio d_z/d_proper w.r.t. 1, with the worst
	``tail_frac`` of pairs up-weighted by ``tail_w`` and the total clamped.
	"""
	ratio  = d_z / (d_proper + 1e-8)
	sq_err = (ratio - 1.0)**2
	k      = max(1, int(len(sq_err) * tail_frac))
	return torch.clamp(sq_err.mean() + tail_w * torch.topk(sq_err, k=k).values.mean(),
	                   max=clamp)

def log_distance_loss(d_z, d_true):
	"""Squared difference of log distances — the long-range (global) loss."""
	return ((torch.log(d_z + 1e-6) - torch.log(d_true + 1e-6))**2).mean()

####################################################################################################################

def scale_percentiles(flow, x1, x2, d_true, n=500):
	"""
	Sample ``n`` pairs and return (median, p25, p10) of the ratio
	d_z / d_proper. Inputs ``x1, x2`` are flow inputs (already lifted if
	an embedding is used).
	"""
	idx = torch.randint(0, len(d_true), (min(n, len(d_true)),))
	with torch.no_grad():
		z1, _ = flow(x1[idx]); z2, _ = flow(x2[idx])
		d_z   = torch.norm(z1 - z2, dim=1).numpy()
	ratios = d_z / (d_true[idx].numpy() + 1e-8)
	return (float(np.median(ratios)),
	        float(np.percentile(ratios, 25)),
	        float(np.percentile(ratios, 10)))

def global_correlation(flow, metric, data, n_pairs=200, phi_col=None,
                       embed_fn=None, seed=None):
	"""
	Pearson correlation between d_proper and d_z on fresh random pairs — a
	cheap global-health check logged during training.

	Parameters
	----------
		embed_fn: callable
			Maps a theta tensor to the flow input.
			Defaults to identity.
	"""
	from scipy.stats import pearsonr
	from .flow import wrap_phi
	rng  = np.random.default_rng(seed)
	idx1 = rng.integers(0, len(data), n_pairs)
	idx2 = rng.integers(0, len(data), n_pairs)
	t1   = torch.tensor(data[idx1], dtype=torch.float32)
	t2   = torch.tensor(data[idx2], dtype=torch.float32)
	if phi_col is not None:
		t2 = wrap_phi(t1, t2, phi_col)
	d_true = metric.proper_distance(t1, t2).numpy()
	if embed_fn is None: embed_fn = lambda t: t
	with torch.no_grad():
		z1, _ = flow(embed_fn(t1)); z2, _ = flow(embed_fn(t2))
	d_z  = torch.norm(z1 - z2, dim=1).numpy()
	mask = d_true > 0
	return float(pearsonr(d_true[mask], d_z[mask])[0])
