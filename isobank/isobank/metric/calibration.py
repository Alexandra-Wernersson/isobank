"""
isobank.calibration
===================

Calibration of the placement radius ``r_target``.

The naive metric radius for a maximum mismatch mu is r_metric = sqrt(2 mu)
(second-order expansion of the match). In practice the quadratic (hessian)
metric under- or over-states the TRUE waveform mismatch at finite
separations, so we calibrate: sample metric-guided pairs at a range of
metric separations, compute the TRUE match for each, bin mean mismatch
against mean proper distance, and interpolate to find the proper distance
at which the mean TRUE mismatch equals mu. That distance is ``r_target``.

Typical outcomes: r_target / r_metric ~ 1.2 for the 4D HM space,
~ 1.5 for the 2D HM space — i.e. the hessian metric is pessimistic and
naive placement would over-populate the bank.

MEAN vs QUANTILE calibration. By default r_target is the distance at which
the MEAN true mismatch equals mu. That is only safe when the relation
between proper distance and true mismatch is homogeneous across the space.
It is not in wide mass ranges: in the aligned-spin O4 space (M in [2, 400])
pairs placed at exactly r_target have mean mismatch 0.037 at M < 10 but
0.027 at M > 80. The means straddle mu, and because the mismatch
distribution at fixed distance is narrow, that small shift takes the
fraction of pairs meeting FF >= 0.97 from 88% at high mass to 0% at low
mass. Pass ``quantile=0.9`` to instead solve for the distance at which 90%
of pairs satisfy mismatch <= mu. This over-covers the benign regions, but
in exactly the spaces that need it those regions carry almost no metric
volume (M > 80 is 29% of the coordinate volume but 0.4% of the metric
volume), so the extra templates are nearly free.
"""

import sys
import numpy as np
import torch

__all__ = ['calibrate_r_target']

####################################################################################################################

def calibrate_r_target(metric, lo, hi, mu=0.03, n_ref=400, quantile=None,
                       fracs=None, seed=42, valid_fn=None, verbose=True):
	"""
	Calibrate the placement radius against TRUE matches.

	Parameters
	----------
		metric: :class:`isobank.metric.cbc_metric`
			The metric object (also provides the true match)
		lo, hi: :class:`~numpy:numpy.ndarray`
			Coordinate boundaries, shape (D,)
		mu: float
			Maximum tolerated mismatch (1 - minimum match)
		n_ref: int
			Number of reference (anchor) points
		quantile: float
			If None (default), r_target is where the MEAN mismatch equals mu.
			If set (e.g. 0.9), r_target is where this FRACTION of pairs
			satisfies mismatch <= mu. Use in spaces where the
			distance-mismatch relation varies strongly across the domain
			(see the module docstring).
		fracs: :class:`~numpy:numpy.ndarray`
			Fractions of r_metric to probe; default linspace(0.1, 3.0, 15)
		valid_fn: callable
			Optional validity check theta -> bool applied to anchors and
			endpoints (e.g. m1 >= m2)

	Returns
	-------
		r_target: float
			Calibrated placement radius (proper distance at mean mismatch mu)
		diag: dict
			Diagnostics: d_proper, mismatches, bin_dist, bin_mean, r_metric
	"""
	D = len(lo)
	r_metric = np.sqrt(2 * mu)
	if fracs is None:
		fracs = np.linspace(0.1, 3.0, 15)
	rng = np.random.default_rng(seed)

		#reference points
	ref_pts = []
	while len(ref_pts) < n_ref:
		cand = rng.uniform(lo, hi, D).astype(np.float32)
		if valid_fn is None or valid_fn(cand):
			ref_pts.append(cand)
	ref_pts = np.array(ref_pts)

	if verbose:
		print(f"Computing metric at {n_ref} reference points...")
		sys.stdout.flush()
	g_ref = metric.get_metric(torch.tensor(ref_pts), metric_type='hessian')

		#drop unstable (non positive definite) anchors
	stable = np.array([np.linalg.eigvalsh(g).min() > 0 for g in g_ref])
	if verbose: print(f"  Stable: {stable.sum()} / {n_ref}")
	ref_pts, g_ref = ref_pts[stable], g_ref[stable]

		#metric-guided pairs at each frac
	pts1, pts2, d_metric = [], [], []
	for i in range(len(ref_pts)):
		g = g_ref[i]
		for frac in fracs:
			v  = rng.normal(0, 1, D)
			gv = float(v @ g @ v)
			if gv < 1e-20: continue
			v  = v * (frac * r_metric) / np.sqrt(gv)
			t2 = ref_pts[i] + v
			if np.all(t2 >= lo) and np.all(t2 <= hi) and \
			   (valid_fn is None or valid_fn(t2)):
				pts1.append(ref_pts[i])
				pts2.append(t2.astype(np.float32))
				d_metric.append(frac * r_metric)
	if verbose:
		print(f"Generated {len(pts1)} metric-guided pairs.")
		print("Computing proper distances and TRUE matches...")
		sys.stdout.flush()

	t1_t = torch.tensor(np.array(pts1, dtype=np.float32))
	t2_t = torch.tensor(np.array(pts2, dtype=np.float32))
	d_metric = np.array(d_metric)

	with torch.no_grad():
		d_proper   = metric.proper_distance(t1_t, t2_t).numpy()
		matches    = np.asarray(metric.match(t1_t, t2_t))
	mismatches = 1.0 - matches

		#bin mean mismatch vs mean proper distance
	bins    = np.linspace(0, fracs.max() * r_metric * 1.05, len(fracs) + 2)
	bin_idx = np.digitize(d_metric, bins)
	bin_mean, bin_dist = [], []
	for k in range(1, len(bins)):
		sel = bin_idx == k
		if sel.sum() < 5: continue
		bin_mean.append(mismatches[sel].mean())
		bin_dist.append(d_proper[sel].mean())
		if verbose:
			print(f"  bin {k:2d}: n={sel.sum():4d}  mean_mismatch={mismatches[sel].mean():.4f}"
			      f"  mean_d_proper={d_proper[sel].mean():.4f}")
	bin_mean, bin_dist = np.array(bin_mean), np.array(bin_dist)

	r_mean = float(np.interp(mu, bin_mean, bin_dist)) \
		if bin_mean.min() < mu < bin_mean.max() else r_metric
	if r_mean == r_metric:
		import warnings
		warnings.warn(f"mismatch range [{bin_mean.min():.4f}, {bin_mean.max():.4f}] "
		              f"does not bracket mu={mu}; falling back to r_metric")

	q_dist = q_pass = None
	if quantile is not None:
			#bin on d_proper (that is what we are solving for) and track the
			#FRACTION of pairs meeting the mismatch budget in each bin
		edges = np.quantile(d_proper, np.linspace(0, 1, len(fracs) + 2))
		edges = np.unique(edges)
		q_dist, q_pass = [], []
		for k in range(len(edges) - 1):
			sel = (d_proper >= edges[k]) & (d_proper < edges[k + 1])
			if sel.sum() < 10: continue
			q_dist.append(d_proper[sel].mean())
			q_pass.append((mismatches[sel] <= mu).mean())
			if verbose:
				print(f"  qbin {k:2d}: n={sel.sum():4d}  d_proper={q_dist[-1]:.4f}"
				      f"  frac(mismatch<={mu})={q_pass[-1]:.3f}")
		q_dist, q_pass = np.array(q_dist), np.array(q_pass)
			#Solve in DISTANCE order, not by sorting on the pass fraction:
			#the pass fraction saturates at 1.0 (and at 0.0), so it has ties,
			#and np.interp on a non-monotonic x silently returns nonsense —
			#with three tied 1.0 bins a stable sort picks the SMALLEST
			#distance and the radius comes out ~2.5x too small.
			#Walk outwards and take the first crossing below the quantile.
		if q_pass.max() >= quantile > q_pass.min():
			k = int(np.argmax(q_pass < quantile))      # first bin below
			if k == 0:
				r_target = float(q_dist[0])
			else:
				p_hi, p_lo = q_pass[k - 1], q_pass[k]  # p_hi >= quantile > p_lo
				w = (p_hi - quantile) / max(p_hi - p_lo, 1e-12)
				r_target = float(q_dist[k - 1] + w * (q_dist[k] - q_dist[k - 1]))
		else:
			import warnings
			warnings.warn(f"pass-fraction range [{q_pass.min():.3f}, "
			              f"{q_pass.max():.3f}] does not bracket quantile="
			              f"{quantile}; falling back to the mean calibration")
			r_target = r_mean
	else:
		r_target = r_mean

	if verbose:
		print(f"\n  Calibrated r_target : {r_target:.5f}"
		      + (f"  (quantile {quantile})" if quantile is not None else "  (mean)"))
		if quantile is not None:
			print(f"  mean-calibrated     : {r_mean:.5f}"
			      f"   -> quantile is {r_target/r_mean:.3f}x, "
			      f"{(r_mean/r_target)**len(lo):.2f}x the templates")
		print(f"  r_metric            : {r_metric:.5f}")
		print(f"  Ratio               : {r_target/r_metric:.4f}")

	diag = dict(d_proper=d_proper, mismatches=mismatches, d_metric=d_metric,
	            bin_dist=bin_dist, bin_mean=bin_mean, r_metric=r_metric,
	            r_mean=r_mean, q_dist=q_dist, q_pass=q_pass,
	            ref_pts=np.array(pts1))
	return r_target, diag
