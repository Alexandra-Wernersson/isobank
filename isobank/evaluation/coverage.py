"""
isobank.coverage
================

TRUE-match coverage validation of a template bank.

For each injection drawn uniformly in the physical space, find the K nearest
templates by z-distance (cKDTree) and compute the true waveform match against
each; the fitting factor is the best of the K. Coverage is the fraction of
injections with FF >= 1 - mu.

Always validate with TRUE matches: the proper distance proxy can overstate
coverage by ~10 percentage points in the worst regions.
"""

import sys
import numpy as np
import torch
from scipy.spatial import cKDTree

__all__ = ['coverage_check']

####################################################################################################################

def coverage_check(metric, flow, templates, lo, hi, embed_fn=None,
                   valid_fn=None, n_inj=2000, k_neighbours=5, mu=0.03,
                   seed=99, verbose=True):
	"""
	TRUE-match coverage check.

	Parameters
	----------
		metric: :class:`isobank.metric.cbc_metric`
			Provides the true match
		flow: :class:`isobank.flow.IsometricFlow`
			Trained flow used for nearest-template queries
		templates: :class:`~numpy:numpy.ndarray`
			shape (N, D) - the bank, in physical coordinates
		embed_fn: callable
			Maps theta tensor to flow input (identity if None)
		valid_fn: callable
			Optional validity mask over an injection batch
		n_inj: int
			Number of injections
		k_neighbours: int
			Number of z-space nearest templates to try per injection
		mu: float
			Maximum mismatch; coverage threshold is FF >= 1 - mu

	Returns
	-------
		best_matches: :class:`~numpy:numpy.ndarray`
			shape (n_inj,) - fitting factor of each injection
		signals: :class:`~numpy:numpy.ndarray`
			shape (n_inj, D) - the injections
	"""
	if embed_fn is None: embed_fn = lambda t: t
	D   = len(lo)
	rng = np.random.default_rng(seed)

	with torch.no_grad():
		z_templates, _ = flow(embed_fn(torch.from_numpy(templates.astype(np.float32))))
	tree = cKDTree(z_templates.numpy())

		#draw injections
	signals = []
	while len(signals) < n_inj:
		cand = rng.uniform(lo, hi, (n_inj, D)).astype(np.float32)
		if valid_fn is not None:
			cand = cand[valid_fn(cand)]
		signals.extend(cand)
	signals = np.array(signals[:n_inj])

	with torch.no_grad():
		z_signals, _ = flow(embed_fn(torch.from_numpy(signals)))
	z_signals = z_signals.numpy()

	if verbose:
		print(f"Computing TRUE matches for {n_inj} injections "
		      f"(K={k_neighbours} neighbours)...")
		sys.stdout.flush()

	best_matches = np.zeros(n_inj)
	for i in range(n_inj):
		_, idxs = tree.query(z_signals[i], k=min(k_neighbours, len(templates)))
		idxs    = np.atleast_1d(idxs)
		sig_t   = torch.from_numpy(signals[i:i+1])
		tmpl_t  = torch.from_numpy(templates[idxs].reshape(-1, D))
		with torch.no_grad():
			m = np.asarray(metric.match(sig_t.expand(len(idxs), -1), tmpl_t))
		best_matches[i] = m.max()

		if verbose and (i + 1) % max(1, n_inj // 10) == 0:
			cov = (best_matches[:i+1] >= 1-mu).mean() * 100
			print(f"  {i+1:5d}/{n_inj}  running coverage (FF>={1-mu}) = {cov:.1f}%")
			sys.stdout.flush()

	if verbose:
		thr = 1.0 - mu
		print(f"\n{'='*60}")
		print(f"Coverage FF >= {thr:.2f} : {(best_matches >= thr).mean()*100:.1f}%  (N={n_inj})")
		print(f"Coverage FF >= 0.95  : {(best_matches >= 0.95).mean()*100:.1f}%")
		print(f"Median FF            : {np.median(best_matches):.4f}")
		print(f"p10 FF               : {np.percentile(best_matches, 10):.4f}")
		print(f"Min FF               : {best_matches.min():.4f}")
		print(f"{'='*60}")

	return best_matches, signals
