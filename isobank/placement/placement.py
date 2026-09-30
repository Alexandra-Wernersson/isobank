"""
isobank.placement
=================

Greedy template placement in the latent space of a trained isometric flow.

Candidates are drawn uniformly in the physical space, mapped through the
flow, and accepted if their Euclidean z-distance to every accepted template
exceeds the placement radius. Scales to multi-million-template banks
(low-mass spaces genuinely need them — the eccentric 5D BNS/NSBH bank is
~4M templates):

- candidates are tested against the bank with ONE vectorized, multi-threaded
  ``cKDTree.query`` per batch (bank tree + a small tree over the pending
  buffer), leaving only the tiny intra-batch dedup to a Python loop;
- the bank tree is rebuilt on a geometric schedule (when the pending buffer
  reaches a fraction of the bank size), so the number of O(N log N) rebuilds
  grows logarithmically with the final bank size instead of linearly;
- the bank is checkpointed periodically and placement can resume from a
  saved (partial) bank — a wall-clock timeout costs at most
  ``partial_every`` batches of work.

**Placement radius**: use ``r_placement = p10 * r_target`` where p10 is the
flow's held-out 10th-percentile ratio at checkpoint selection (take
``min(p10, p10_edge)`` when an edge monitor was tracked). The flow
under-expands in ~10% of the space; templates placed at r_target in z-space
would there be more than r_target apart in proper distance, leaving coverage
holes. Shrinking the radius by p10 restores proper-distance spacing at most
r_target everywhere, at the cost of ~(1/p10)^D more templates.
(Validated on the 2D HM bank: p10=0.89, coverage at FF>=0.97 went from
96.8% to 98.6% for 28% more templates.)
"""

import sys
import numpy as np
import torch
from scipy.spatial import cKDTree

__all__ = ['greedy_placement']

####################################################################################################################

def _flow_z(flow, embed_fn, theta_np, chunk=200_000, with_logdet=False):
	"""Map a (N, D) numpy array through the flow in chunks.

	``with_logdet`` also returns the log-Jacobian, which for an isometric
	flow is log(sqrt|g|) — i.e. the local metric-volume density."""
	outs, lds = [], []
	for i in range(0, len(theta_np), chunk):
		with torch.no_grad():
			z, ld = flow(embed_fn(torch.from_numpy(theta_np[i:i+chunk])))
		outs.append(z.numpy())
		if with_logdet: lds.append(ld.numpy())
	if with_logdet:
		return np.vstack(outs), np.concatenate(lds)
	return np.vstack(outs)

def greedy_placement(flow, r_placement, lo, hi, embed_fn=None, valid_fn=None,
                     batch=50000, max_batches=5000, pending_max=2000,
                     rebuild_frac=0.05, stop_acceptance=0.01, seed=0,
                     min_ess_frac=0.05, proposal_fn=None, stop_on='volume',
                     initial_templates=None, partial_save=None,
                     partial_every=200, verbose=True):
	"""
	Greedy ball-packing bank placement.

	Parameters
	----------
		flow: :class:`isobank.flow.IsometricFlow`
			Trained flow (eval mode)
		r_placement: float
			Placement radius in z-space (recommended: p10 * r_target)
		lo, hi: :class:`~numpy:numpy.ndarray`
			Physical-space boundaries for candidate sampling
		embed_fn: callable
			Maps a (N, D) theta tensor to the flow input. Defaults to
			identity.
		valid_fn: callable
			Optional validity mask over a candidate batch,
			(N, D) array -> (N,) bool (e.g. m1 >= m2)
		pending_max, rebuild_frac: int, float
			The bank tree is rebuilt when the pending buffer exceeds
			``max(pending_max, rebuild_frac * bank_size)`` — a geometric
			schedule that keeps the total rebuild cost near O(N log^2 N)
		stop_acceptance: float
			Stop when the uncovered METRIC-VOLUME fraction of the last batch
			drops below this. The fraction is estimated by weighting uniform
			candidates with the flow's Jacobian (|det J| = sqrt|g| for an
			isometric flow), so it measures the volume that actually governs
			coverage. An unweighted count would measure uncovered COORDINATE
			volume, which badly underestimates the remaining work in spaces
			where the metric volume is concentrated (e.g. wide mass ranges).
		min_ess_frac: float
			Warn once if the effective sample size of the metric-volume
			weights falls below this fraction of the batch, which means the
			stopping criterion cannot be trusted in this space
		proposal_fn: callable
			Optional biased proposal ``(rng, n) -> (candidates, logq)`` where
			``candidates`` is (n, D) and ``logq`` is the log proposal density
			(up to an additive constant) at each candidate. Draw proportional
			to sqrt|g| (uniform-in-latent-space, e.g. log-uniform in the mass
			columns) so metrically dense regions fill at the same rate as the
			rest — otherwise a thin high-volume sliver fills far too slowly and
			the metric-weighted ESS collapses. The acceptance rule and hence
			the final packing are unchanged; only the fill order and the
			stopping estimate's ESS improve. Defaults to uniform on [lo, hi].
		initial_templates: :class:`~numpy:numpy.ndarray`
			Optional (N, D) bank to resume from (e.g. a partial save)
		partial_save: str
			If given, the bank is saved to this path every ``partial_every``
			batches (and at the end), so a timeout loses little work

	Returns
	-------
		templates: :class:`~numpy:numpy.ndarray`
			shape (N_templates, D) - physical-space template coordinates
	"""
	if embed_fn is None: embed_fn = lambda t: t
	rng = np.random.default_rng(seed)
	D   = len(lo)

	bank_theta, bank_z = [], []
	if initial_templates is not None and len(initial_templates) > 0:
		init = np.asarray(initial_templates, dtype=np.float32)
		bank_theta = list(init)
		bank_z     = list(_flow_z(flow, embed_fn, init))
		if verbose:
			print(f"Resuming from {len(bank_theta):,} existing templates.")

	tree = cKDTree(np.array(bank_z)) if bank_z else None
	pending_z   = []    # accepted since last tree rebuild
	n_evaluated = 0
	_ess_warned = False

	if verbose:
		print(f"Starting greedy placement  (r_placement={r_placement:.5f}, "
		      f"batch={batch:,})...")
		sys.stdout.flush()

	for batch_idx in range(max_batches):
			#draw candidates, mask invalid, map to z (all vectorized)
		if proposal_fn is None:
			cands = rng.uniform(lo, hi, (batch, D)).astype(np.float32)
			logq  = None
		else:
				#metric-volume-proportional (or any biased) proposals. The bank
				#is unchanged — the acceptance rule is still pure latent
				#distance — only WHERE candidates are drawn changes, which
				#governs how fast each region fills, not the final packing.
			cands, logq = proposal_fn(rng, batch)
			cands = np.asarray(cands, dtype=np.float32)
			logq  = np.asarray(logq, dtype=np.float64)
		if valid_fn is not None:
			m = valid_fn(cands)
			cands = cands[m]
			if logq is not None: logq = logq[m]
		if len(cands) == 0:
			continue
		z_cands, ld_cands = _flow_z(flow, embed_fn, cands, with_logdet=True)
		n_valid = len(cands)
		n_evaluated += n_valid
			#Metric-volume weights. For an isometric flow |det J| = sqrt|g|, so
			#sqrt|g| = exp(logdet). To estimate the metric volume the candidates
			#represent, each is weighted by sqrt|g| / q(theta), where q is the
			#PROPOSAL density. For uniform proposals q is constant and the
			#weight is just exp(logdet); for biased proposals the log-density
			#logq must be subtracted, else the estimate double-counts the
			#regions the proposal already concentrates on.
		logw = ld_cands if logq is None else (ld_cands - logq)
		w_all = np.exp(logw - logw.max())
		w_total = w_all.sum()
			#The weighted estimate is only as good as its effective sample
			#size. Uniform-in-coordinates proposals resolve sqrt|g| well when
			#the metric volume is spread out (aligned-spin 3D: ESS ~29%) but
			#badly when it concentrates in a thin sliver (eccentric 5D: ESS
			#~1%, where 1000 of 300,000 samples carry half the volume). Below
			#the threshold the stopping criterion is NOT trustworthy and the
			#bank must be validated against an explicit coverage test.
		ess_frac = float(w_total**2 / (np.sum(w_all**2) * max(n_valid, 1)))
		if ess_frac < min_ess_frac and not _ess_warned:
			_ess_warned = True
			print(f"WARNING: metric-volume weights have effective sample size "
			      f"{100*ess_frac:.1f}% of the batch (< {100*min_ess_frac:.0f}%).\n"
			      f"         The stopping criterion is unreliable in this space "
			      f"— validate the bank with an explicit coverage test and do "
			      f"not trust convergence alone.")
			sys.stdout.flush()

			#vectorized pre-filter against the bank tree
		if tree is not None:
			dd, _ = tree.query(z_cands, k=1, workers=-1)
			keep  = dd > r_placement
			cands, z_cands, w_all = cands[keep], z_cands[keep], w_all[keep]

			#vectorized pre-filter against the pending buffer
		if len(pending_z) > 0 and len(z_cands) > 0:
			ptree = cKDTree(np.array(pending_z))
			dd, _ = ptree.query(z_cands, k=1, workers=-1)
			keep  = dd > r_placement
			cands, z_cands, w_all = cands[keep], z_cands[keep], w_all[keep]

			#uncovered METRIC-volume fraction: the quantity that actually
			#governs coverage. Measured BEFORE intra-batch dedup (dedup drops
			#points that are genuinely uncovered by the bank).
		w_uncovered = w_all.sum()

			#intra-batch dedup: survivors only conflict with each other now.
			#Brute-force O(n^2) scan — only expensive in the first few batches
			#when the bank is empty and most candidates survive the prefilters.
		new_z = np.empty((len(z_cands), z_cands.shape[1]), dtype=np.float32) \
			if len(z_cands) else np.empty((0, D), dtype=np.float32)
		n_new = 0
		for j in range(len(cands)):
			zc = z_cands[j]
			if n_new > 0 and \
			   np.linalg.norm(new_z[:n_new] - zc, axis=1).min() <= r_placement:
				continue
			new_z[n_new] = zc; n_new += 1
			bank_theta.append(cands[j])
			bank_z.append(zc)
			pending_z.append(zc)

			#geometric rebuild schedule
		threshold = max(pending_max, int(rebuild_frac * len(bank_theta)))
		if len(pending_z) >= threshold:
			tree = cKDTree(np.array(bank_z))
			pending_z = []

			#Two estimators. The metric-volume-weighted one is correct:
			#the raw count is an estimate of uncovered COORDINATE volume,
			#which in spaces with strong metric-volume heterogeneity can be
			#many times smaller and causes premature stopping.
		acc_batch = w_uncovered / max(w_total, 1e-300)
		acc_coord = n_new / max(n_valid, 1)
			#which acceptance governs stopping. 'volume' (metric-volume
			#weighted) is the default and the physically correct measure of
			#uncovered coverage; 'coord' (raw coordinate acceptance) matches the
			#convergence/tolerance criterion of a classic stochastic bank, which
			#proposes uniformly in coordinates and stops on the accepted fraction.
		acc_stop = acc_batch if stop_on == 'volume' else acc_coord

		if verbose and batch_idx % 25 == 0:
			print(f"  Batch {batch_idx:4d} | evaluated={n_evaluated:,}  "
			      f"bank={len(bank_theta):,}  uncovered_vol={acc_batch:.4f}  "
			      f"(coord={acc_coord:.4f})  [stop_on={stop_on}]")
			sys.stdout.flush()

		if partial_save and batch_idx > 0 and batch_idx % partial_every == 0:
			np.save(partial_save, np.array(bank_theta, dtype=np.float32))
			if verbose:
				print(f"  [partial save: {len(bank_theta):,} templates -> {partial_save}]")
				sys.stdout.flush()

		if acc_stop < stop_acceptance:
			converged = True
			if verbose:
				label = 'metric volume' if stop_on == 'volume' else 'coordinate acceptance'
				print(f"  Uncovered {label} {acc_stop:.4f} < {stop_acceptance} "
				      f"— converged, stopping.  (vol={acc_batch:.4f} coord={acc_coord:.4f})")
			break
	else:
		converged = False

	templates = np.array(bank_theta, dtype=np.float32)
	if partial_save:
		np.save(partial_save, templates)
	if verbose:
		if converged:
			print(f"Bank complete: {len(templates):,} templates")
		else:
				#Do NOT report this as a finished bank. Exhausting max_batches
				#leaves an UNSATURATED bank that looks complete but under-covers
				#(seen on the wide aligned-spin space: 5.3M templates truncated
				#at uncovered_vol=0.037 gave 75% coverage, not the ~97% the
				#threshold was set for). Re-run with a larger max_batches, or
				#resume from the partial save.
			print(f"WARNING: hit max_batches={max_batches} with uncovered "
			      f"{'metric volume' if stop_on=='volume' else 'coordinate acceptance'} "
			      f"{acc_stop:.4f} still above stop_acceptance={stop_acceptance}.\n"
			      f"         The bank is UNSATURATED and will under-cover. "
			      f"Increase max_batches or resume from the partial save.")
	return templates
