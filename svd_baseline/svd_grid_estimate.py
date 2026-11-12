"""
How well would a plain SVD grid do on the isobank spaces (no autoencoder, no flow)?

Geometric placement in the phase-SVD coordinates, as in Roulet et al. and in Cabass et al.'s linear baseline:
a cubic grid of spacing Delta in the top-k coefficients, other coefficients set to zero.

Coverage does not need the bank to be built: snapping an injection's coefficients to the grid gives its template.
Overlap is the phase overlap with one amplitude profile per bank, without further time/phase maximisation,
so it is a lower bound on the fitting factor of that template.

Bank size is bounded from above by (convex hull volume of the coefficients) / Delta^k. The hull is built from
training waveforms plus the corners/edges of the parameter box; the true region is not convex, so the true
size is smaller. Upper bound only.

Usage: python svd_grid_estimate.py <ecc5d|al3d>
"""
import sys, os, json
import numpy as np
from multiprocessing import Pool
from scipy.spatial import ConvexHull
import svd_feasibility as F          # reads sys.argv[1]

SPACE, S = F.SPACE, F.S
N_TRAIN = 6000
DELTAS = [.15, .2, .25, .283, .32, .36, .4, .5]
KS = [2, 3, 4, 5]

def injections():
	if SPACE == 'ecc5d':            # the paper's injections (seed 4002), first 4000
		rng = np.random.default_rng(4002); n = 10000; x = rng.uniform([1, 1, -.9, -.05, 1e-5], [7, 3, .9, .05, .15], (4 * n, 5)).astype(np.float32)
		return x[x[:, 0] >= x[:, 1]][:4000].astype(float)
	x = np.random.default_rng(4242).uniform([30, 1, -.99], [50, 5, .99], (5000, 3)).astype(np.float32).astype(float)   # first 5000 of the paper's
	m1 = x[:, 0] * x[:, 1] / (1 + x[:, 1]); m2 = x[:, 0] / (1 + x[:, 1])
	return np.c_[m1, m2, x[:, 2], x[:, 2], 0 * m1]

def box_points():
	import itertools
	if SPACE == 'ecc5d':
		g = itertools.product([1, 2, 3, 5, 7], [1, 1.5, 2, 3], [-.9, 0, .9], [-.05, .05], [1e-5, .075, .15])
		return np.array([p for p in g if p[0] >= p[1]], float)
	x = np.array(list(itertools.product([30, 35, 40, 45, 50], [1, 2, 3, 4, 5], [-.99, -.5, 0, .5, .99])), float)
	m1 = x[:, 0] * x[:, 1] / (1 + x[:, 1]); m2 = x[:, 0] / (1 + x[:, 1])
	return np.c_[m1, m2, x[:, 2], x[:, 2], 0 * m1]

def main():
	f_psd, s = np.loadtxt(S['psd']).T[:2]
	psd = np.interp(F.GRID, f_psd, s ** 2 if S['asd'] else s)
	tr, inj, box = F.draw(N_TRAIN, 21), injections(), box_points()
	X = np.vstack([tr, box, inj]); n1, n2 = len(tr), len(box)
	with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
		out = pool.map(F.waveform, X, chunksize=16)
	A = np.array([o[0] for o in out]); P = np.array([o[1] for o in out])
	W = A / np.sqrt(psd); W /= np.sqrt((W ** 2).sum(1, keepdims=True))
	w = W[:n1].mean(0); w /= np.sqrt((w ** 2).sum()); mask = w > 1e-12
	Q, _ = np.linalg.qr(np.c_[np.ones_like(F.GRID), F.GRID - F.GRID.mean()] * w[:, None])
	mean = P[:n1].mean(0)
	for _ in range(3):
		R = (P - mean) * w; R -= (R @ Q) @ Q.T
		mean = mean + R[:n1].mean(0) / np.where(w > 1e-12, w, np.inf)
	R = (P - mean) * w; R -= (R @ Q) @ Q.T
	_, sv, Vt = np.linalg.svd(R[:n1], full_matrices=False)
	C = R @ Vt[:8].T
	np.savez(os.path.join(F.OUT, f'svd_coeffs_{SPACE}.npz'), C=C, X=X, n1=n1, n2=n2, sv=sv[:12])
	c_hull, c_inj, r_inj = C[:n1 + n2], C[n1 + n2:], R[n1 + n2:]; W_inj = W[n1 + n2:]
	print('coefficient std', np.round(C[:n1].std(0), 3), 'range', np.round(np.ptp(c_hull, 0), 3), flush=True)

	def overlaps(k, delta):
		snap = np.rint(c_inj[:, :k] / delta) * delta
		rec = snap @ Vt[:k]
		d = np.zeros_like(rec); d[:, mask] = (rec - r_inj)[:, mask] / w[mask]
		amp = (W_inj * w * np.cos(d)).sum(1)          # signal amplitude x template amplitude x cos(phase error)
		phase_only = (w ** 2 * np.cos(d)).sum(1)
		return amp, phase_only

	rows = []
	for k in KS:
		try: vol = ConvexHull(c_hull[:, :k]).volume
		except Exception as e:                   # Qhull fails on the nearly degenerate ecc5d coefficients; svd_grid_occupancy.py counts cells instead
			print(f'convex hull failed for k={k}: {e.__class__.__name__}', flush=True); vol = np.nan
		for delta in DELTAS:
			amp, ph = overlaps(k, delta)
			row = dict(k=k, delta=delta, n_hull_upper=float(vol / delta ** k), cov_amp=float((amp >= .97).mean()), cov_phase=float((ph >= .97).mean()),
			           worst=float(amp.min()), p1=float(np.percentile(amp, 1)))
			rows.append(row); print(json.dumps(row), flush=True)
	json.dump(dict(space=SPACE, rows=rows, sv=[float(x) for x in sv[:12]]), open(os.path.join(F.OUT, f'svd_grid_{SPACE}.json'), 'w'), indent=1)

if __name__ == '__main__':
	main()
