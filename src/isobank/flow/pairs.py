"""
isobank.pairs
=============

Training-pair generation for the isometric flow. Three kinds of pairs are
used, all with TRUE proper distances from :meth:`cbc_metric.proper_distance`:

- **long-range pairs**: random point pairs, teach the global layout
  (used with a log-distance loss).
- **local pairs**: metric-guided pairs at fractions [~0.05, 1.5] of r_target,
  teach the local isometry (ratio loss). These MUST be generated
  metric-guided: random pairs essentially never land within a few r_target
  of each other in regions where the metric is small (e.g. high total mass,
  where the waveform has few cycles in band), silently starving the local
  loss of training signal.
- **boundary pairs**: metric-guided pairs filtered to [0.5, 1.5] x r_target,
  the scale that actually matters for template placement. Also used (with a
  held-out seed) as the monitoring set for checkpoint selection.
"""

import numpy as np
import torch

from .flow import wrap_phi, wrap_phi_np

__all__ = ['long_range_pairs', 'metric_guided_pairs', 'filter_by_distance',
           'degenerate_direction_pairs']

####################################################################################################################

def long_range_pairs(data, metric, n_pairs, d_max=5.0, seed=42, phi_col=None):
	"""
	Random long-range pairs with true proper distances.

	Parameters
	----------
		data: :class:`~numpy:numpy.ndarray`
			shape (N, D) - point cloud to draw pairs from
		metric: :class:`isobank.metric.cbc_metric`
			Metric object (provides proper_distance)
		n_pairs: int
			Number of pairs to draw (before the d < d_max cut)
		d_max: float
			Drop pairs with proper distance above this, or ``None`` to keep
			all finite pairs. IMPORTANT: pick relative to the proper-distance
			diameter of YOUR space. The long-range pairs are what bootstraps
			the flow's global scale — if d_max cuts away the bulk of the
			distance distribution, the ratio losses saturate their clamp at
			init (zero gradient) and training never starts. For spaces
			hundreds of units across (e.g. eccentric low-mass spaces), use
			``None``; a small d_max is only appropriate when the whole space
			is a few tens of units wide.
		phi_col: int
			If given, wrap this periodic column to the shortest path before
			computing distances

	Returns
	-------
		t1, t2: :class:`torch.Tensor`
			shape (M, D) - pair endpoints
		d_true: :class:`torch.Tensor`
			shape (M,) - proper distances
	"""
	rng   = np.random.default_rng(seed)
	pidx1 = rng.integers(0, len(data), n_pairs)
	pidx2 = rng.integers(0, len(data), n_pairs)
	same  = pidx1 == pidx2; pidx2[same] = (pidx2[same] + 1) % len(data)

	t1 = torch.from_numpy(data[pidx1]).float()
	t2 = torch.from_numpy(data[pidx2]).float()
	if phi_col is not None:
		t2 = wrap_phi(t1, t2, phi_col)

	with torch.no_grad():
		d_true = metric.proper_distance(t1, t2)
	keep = torch.isfinite(d_true) & (d_true > 1e-6)
	if d_max is not None:
		keep &= d_true < d_max
	return t1[keep], t2[keep], d_true[keep]

####################################################################################################################

def metric_guided_pairs(data, g_all, r_target, n_pairs, frac_range,
                        lo, hi, seed, phi_col=None, valid_fn=None,
                        max_iter=8_000_000):
	"""
	Metric-guided pairs: from a random anchor, step in a random direction v
	scaled so the METRIC length of the step is ``frac * r_target`` with
	``frac ~ U(frac_range)``. This concentrates pairs at placement-relevant
	separations everywhere in the space, including regions where the metric
	is tiny and random pairs are never local.

	Parameters
	----------
		data: :class:`~numpy:numpy.ndarray`
			shape (N, D) - anchors are drawn from here
		g_all: :class:`~numpy:numpy.ndarray`
			shape (N, D, D) - metric evaluated at each anchor
			(from ``metric.get_metric(data, metric_type='hessian')``)
		r_target: float
			Calibrated placement radius (see :mod:`isobank.calibration`)
		n_pairs: int
			Target number of pairs
		frac_range: tuple
			(frac_lo, frac_hi), e.g. (0.05, 1.5) for local pairs,
			(0.3, 2.0) for boundary/monitoring pairs
		lo, hi: :class:`~numpy:numpy.ndarray`
			Coordinate boundaries; candidate endpoints outside are rejected
		phi_col: int
			If given, this column is periodic: the step is wrapped into
			[-pi, pi] instead of bound-checked. Near-degenerate anchors
			(where the metric blows the phi step up) then still yield valid
			pairs with small proper distance.
		valid_fn: callable
			Optional extra validity check theta -> bool (e.g. m1 >= m2)

	Returns
	-------
		t1, t2: :class:`torch.Tensor`
			shape (M, D) - pair endpoints (t2 phi-wrapped w.r.t. t1 if
			``phi_col`` is set)
	"""
	rng = np.random.default_rng(seed)
	D   = data.shape[1]
	frac_lo, frac_hi = frac_range
	t1_list, t2_list = [], []

	check_cols = [j for j in range(D) if j != phi_col]

	for _ in range(max_iter):
		i  = rng.integers(0, len(data))
		g  = g_all[i]; v = rng.normal(0, 1, D)
		gv = float(v @ g @ v)
		if gv < 1e-20: continue
		frac = rng.uniform(frac_lo, frac_hi)
		v    = v * (r_target * frac / np.sqrt(gv))
		t2   = data[i] + v
		if phi_col is not None:
			t2[phi_col] = wrap_phi_np(t2[phi_col])
		if np.any(t2[check_cols] < lo[check_cols]) or np.any(t2[check_cols] > hi[check_cols]):
			continue
		if valid_fn is not None and not valid_fn(t2):
			continue
		t1_list.append(data[i]); t2_list.append(t2.astype(np.float32))
		if len(t1_list) >= n_pairs: break

	t1 = torch.tensor(np.array(t1_list), dtype=torch.float32)
	t2 = torch.tensor(np.array(t2_list), dtype=torch.float32)
	if phi_col is not None:
		t2 = wrap_phi(t1, t2, phi_col)
	return t1, t2

####################################################################################################################

def degenerate_direction_pairs(data, g_all, n_pairs, lo, hi, seed,
                               n_soft=2, mag_range=(0.02, 0.5),
                               valid_fn=None, phi_col=None, max_iter=8_000_000):
	"""
	Pairs that step along the SMALLEST-eigenvalue eigenvector(s) of the local
	metric — the near-degenerate directions.

	A random-direction step (:func:`metric_guided_pairs`) almost never samples
	these directions, because ``v @ g @ v`` for ``v ~ N(0, I)`` is dominated by
	the stiff eigenvalues. The flow therefore never learns to collapse the soft
	directions and over-expands them in latent space, so greedy placement keeps
	physically near-identical templates that differ only along them. On the
	eccentric 5D bank this was measured directly: the median nearest-neighbour
	proper distance was 0.68 r_target (efficient packing ~1.0), a ~2x template
	redundancy, and the redundant pairs differed mainly along a tilted
	eigenvector in the mass plane — NOT along a soft coordinate axis, which is
	why targeting the metric's eigenvectors (not the coordinates) is the fix.

	Each pair steps a LARGE coordinate distance along a soft eigenvector but has
	a SMALL true proper distance. Filtered to the placement band by the caller
	(``filter_by_distance``) and trained with the ratio loss like the local /
	boundary pairs, this teaches the flow to map the soft direction to small
	latent extent, removing the redundancy.

	Parameters
	----------
		data: :class:`~numpy:numpy.ndarray`
			shape (N, D) - anchors are drawn from here
		g_all: :class:`~numpy:numpy.ndarray`
			shape (N, D, D) - metric at each anchor. A negative smallest
			eigenvalue (the hessian is numerically unstable in near-degenerate
			regions) is fine: only the eigenVECTOR is used for the direction,
			and the TRUE proper distance is the training target.
		n_soft: int
			Step along one of the ``n_soft`` smallest-eigenvalue eigenvectors,
			chosen at random per anchor (2 covers both soft directions of the
			eccentric 5D metric; the very smallest is typically the NS-spin
			axis, the second the mass-plane degeneracy that drives redundancy).
		mag_range: tuple
			Step magnitude as a fraction of the DISTANCE-TO-BOUNDARY along the
			chosen eigenvector (not of the box diagonal). This is essential:
			the eigenvectors have wildly different in-box extents — the
			mass-plane degenerate direction hits the narrow secondary-mass
			bound almost immediately, while a fixed diagonal-fraction step
			leaves the box and is rejected, silently starving training of the
			very direction that drives the redundancy. Scaling by the
			per-direction boundary distance samples every eigenvector uniformly.

	Returns
	-------
		t1, t2: :class:`torch.Tensor`
			shape (M, D) - pair endpoints
	"""
	rng = np.random.default_rng(seed)
	D   = data.shape[1]
	lo, hi = np.asarray(lo, float), np.asarray(hi, float)
	check_cols = [j for j in range(D) if j != phi_col]
	t1_list, t2_list = [], []

	def boundary_distance(x, u):
		"""Largest t >= 0 with x + t*u inside [lo, hi] on the checked cols."""
		t = np.inf
		for j in check_cols:
			if abs(u[j]) < 1e-12:
				continue
			bound = hi[j] if u[j] > 0 else lo[j]
			t = min(t, (bound - x[j]) / u[j])
		return max(t, 0.0)

	for _ in range(max_iter):
		i = rng.integers(0, len(data))
			#symmetrize for numerical safety, eigendecompose (ascending)
		g = 0.5 * (np.asarray(g_all[i]) + np.asarray(g_all[i]).T)
		try:
			_, V = np.linalg.eigh(g)
		except np.linalg.LinAlgError:
			continue
		k    = rng.integers(0, min(n_soft, D))
		u    = V[:, k]                                   # k-th smallest eigvec
		u    = u * (1.0 if rng.random() < 0.5 else -1.0) # random orientation
		t_max = boundary_distance(data[i], u)
		if t_max < 1e-6:
			continue
		step = rng.uniform(*mag_range) * t_max * u
		t2   = data[i] + step
		if phi_col is not None:
			t2[phi_col] = wrap_phi_np(t2[phi_col])
		if np.any(t2[check_cols] < lo[check_cols]) or np.any(t2[check_cols] > hi[check_cols]):
			continue
		if valid_fn is not None and not valid_fn(t2):
			continue
		t1_list.append(data[i]); t2_list.append(t2.astype(np.float32))
		if len(t1_list) >= n_pairs:
			break

	t1 = torch.tensor(np.array(t1_list), dtype=torch.float32)
	t2 = torch.tensor(np.array(t2_list), dtype=torch.float32)
	if phi_col is not None:
		t2 = wrap_phi(t1, t2, phi_col)
	return t1, t2

####################################################################################################################

def filter_by_distance(metric, t1, t2, d_min, d_max):
	"""
	Compute proper distances for the given pairs and keep those with
	``d_min < d < d_max``.

	Returns
	-------
		t1, t2: :class:`torch.Tensor`
			Filtered pair endpoints
		d_true: :class:`torch.Tensor`
			Their proper distances
	"""
	with torch.no_grad():
		d_true = metric.proper_distance(t1, t2)
	keep = (d_true > d_min) & (d_true < d_max)
	return t1[keep], t2[keep], d_true[keep]
