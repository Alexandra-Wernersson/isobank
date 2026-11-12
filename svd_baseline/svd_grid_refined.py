"""
Eccentric 5D: bank size and coverage of a plain phase-SVD grid, done carefully.

Differences from svd_grid_occupancy.py, whose counts did not converge in 5D:
  * coverage uses the exact phase overlap, sum w^2 cos(d), not its small-angle form;
  * the regressor is trained with 20,000 extra waveforms drawn uniformly in Mc^(-5/3), the direction of the largest SVD coefficient;
  * cells are counted with samples uniform in Mc^(-5/3) (80%) plus uniform in the box (20%), so that every slab of the first coefficient is sampled;
  * besides the cubic grid, a body-centred cubic lattice is tested on the top 3 coefficients.

The SVD basis is the one from svd_grid_estimate.py (same 6000 training waveforms, seed 21).

Usage: python svd_grid_refined.py ecc5d
"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn
from multiprocessing import Pool
import svd_feasibility as F
import svd_grid_estimate as G

HERE = os.path.dirname(os.path.abspath(__file__))
N_EXTRA = 20000; NS = [10_000_000, 30_000_000, 100_000_000, 200_000_000]; CHUNK = 2_000_000; KMAX = 5
CUBIC = [(k, d) for k in (3, 5) for d in (.32, .34, .36, .38)]; BCC = [.44, .48, .52, .56, .60]
NCPU = int(os.environ.get('SLURM_CPUS_PER_TASK', 8)); torch.set_num_threads(NCPU)

def u_of(m1, m2):
	return (m1 * m2) ** -1. * (m1 + m2) ** (1 / 3)

def bisect(fun, lo, hi, target, n=40):
	"""fun decreasing in x on [lo, hi]; returns x with fun(x) = target, clipped to the interval."""
	lo, hi = lo.copy(), hi.copy()
	for _ in range(n):
		mid = .5 * (lo + hi); go_up = fun(mid) > target
		lo = np.where(go_up, mid, lo); hi = np.where(go_up, hi, mid)
	return .5 * (lo + hi)

def sample_u(n, rng):
	u = rng.uniform(u_of(7., 3.), u_of(1., 1.), n)
	m2_lo = bisect(lambda m: u_of(7. + 0 * m, m), np.full(n, 1.), np.full(n, 3.), u)
	m2_hi = np.clip(u ** -.6 * 2 ** .2, 1., 3.)
	m2 = rng.uniform(m2_lo, np.maximum(m2_lo, m2_hi))
	m1 = bisect(lambda m: u_of(m, m2), m2, np.full(n, 7.), u)
	return np.c_[m1, m2, rng.uniform(-.9, .9, n), rng.uniform(-.05, .05, n), rng.uniform(1e-5, .15, n)]

def sample(n, rng):
	nb = n // 5; x = rng.uniform([1, 1, -.9, -.05, 1e-5], [7, 3, .9, .05, .15], (nb, 5))
	bad = x[:, 0] < x[:, 1]; x[bad, 0], x[bad, 1] = x[bad, 1].copy(), x[bad, 0].copy()
	return np.vstack([sample_u(n - nb, rng), x])

def feats(p, dtype):
	m1, m2, s1, s2, e = [torch.as_tensor(p[:, i], dtype=torch.float64) for i in range(5)]
	mc = (m1 * m2) ** .6 / (m1 + m2) ** .2
	return torch.stack([torch.log(m1), torch.log(m2), s1, s2, e, mc ** (-5 / 3), mc ** -1, m2 / m1], 1).to(dtype)

def snap_bcc(z, a):
	"""Nearest point of the BCC lattice of cubic constant a. Returns the lattice points and integer ids."""
	i0 = np.rint(z / a); i1 = np.rint(z / a - .5)
	p0, p1 = i0 * a, (i1 + .5) * a
	use1 = ((z - p1) ** 2).sum(1) < ((z - p0) ** 2).sum(1)
	p = p0.copy(); p[use1] = p1[use1]
	ids = np.c_[np.where(use1[:, None], i1, i0).astype(np.int64), use1.astype(np.int64)]
	return p, ids

def main():
	f_psd, s = np.loadtxt(F.S['psd']).T[:2]
	psd = np.interp(F.GRID, f_psd, s ** 2 if F.S['asd'] else s)
	tr, box, inj = F.draw(G.N_TRAIN, 21), G.box_points(), G.injections()
	extra = sample_u(N_EXTRA, np.random.default_rng(77))
	X = np.vstack([tr, box, extra, inj]); n1, n2, n3 = len(tr), len(box), len(extra); nfit = n1 + n2 + n3
	with Pool(NCPU) as pool:
		out = pool.map(F.waveform, X, chunksize=16)
	A = np.array([o[0] for o in out]); P = np.array([o[1] for o in out]); del out
	print('waveforms done', time.strftime('%X'), flush=True)
	W = A / np.sqrt(psd); W /= np.sqrt((W ** 2).sum(1, keepdims=True))
	w = W[:n1].mean(0); w /= np.sqrt((w ** 2).sum()); mask = w > 1e-12
	Q, _ = np.linalg.qr(np.c_[np.ones_like(F.GRID), F.GRID - F.GRID.mean()] * w[:, None])
	mean = P[:n1].mean(0)
	for _ in range(3):
		R = (P - mean) * w; R -= (R @ Q) @ Q.T
		mean = mean + R[:n1].mean(0) / np.where(w > 1e-12, w, np.inf)
	R = (P - mean) * w; R -= (R @ Q) @ Q.T
	_, sv, Vt = np.linalg.svd(R[:n1], full_matrices=False)
	C = R @ Vt[:8].T; ci, ri, Wi = C[nfit:], R[nfit:], W[nfit:]
	mc_i = (inj[:, 0] * inj[:, 1]) ** .6 / (inj[:, 0] + inj[:, 1]) ** .2
	print('top sv', np.round(sv[:6], 3), flush=True)

	def coverage(snap):
		rec = snap @ Vt[:snap.shape[1]]
		d = np.zeros_like(rec); d[:, mask] = (rec - ri)[:, mask] / w[mask]
		ov = (w ** 2 * np.cos(d)).sum(1); amp = (Wi * w * np.cos(d)).sum(1)
		return dict(cov=float((amp >= .97).mean()), cov_phase=float((ov >= .97).mean()), p1=float(np.percentile(amp, 1)), worst=float(amp.min()),
		            cov_mc14=float((amp[mc_i < 1.4] >= .97).mean()), cov_mc11=float((amp[mc_i < 1.1] >= .97).mean()))

	Xt = feats(X[:nfit], torch.float64); Ct = torch.tensor(C[:nfit, :KMAX], dtype=torch.float64)
	mx, sx, my, sy = Xt.mean(0), Xt.std(0), Ct.mean(0), Ct.std(0); sx[sx == 0] = 1
	net = nn.Sequential(nn.Linear(8, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, KMAX)).double()
	N_EP = 20000; opt = torch.optim.Adam(net.parameters(), 2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EP)
	xn, yn = (Xt - mx) / sx, (Ct - my) / sy
	for ep in range(N_EP):
		opt.zero_grad(); l = ((net(xn) - yn) ** 2).mean(); l.backward(); opt.step(); sch.step()
		if ep % 4000 == 0: print(ep, float(l), time.strftime('%X'), flush=True)
	net = net.float(); mxf, sxf = mx.float(), sx.float()

	def predict(p):
		with torch.no_grad(): return net((feats(p, torch.float32) - mxf) / sxf).double().numpy() * sy.numpy() + my.numpy()

	err = predict(inj) - ci[:, :KMAX]
	rms = np.sqrt((err ** 2).mean(0)); lo = mc_i < 1.4
	print('regressor rms on injections', np.round(rms, 4), 'max', np.round(np.abs(err).max(0), 3), 'rms Mc<1.4', np.round(np.sqrt((err[lo] ** 2).mean(0)), 4), flush=True)

	keys = [('cubic', k, d) for k, d in CUBIC] + [('bcc', 3, a) for a in BCC]
	cells = {key: [] for key in keys}; counts = {}; done = 0; rng = np.random.default_rng(6)
	for target in NS:
		while done < target:
			z = predict(sample(CHUNK, rng)); done += CHUNK
			for key in keys:
				scheme, k, d = key
				ids = np.rint(z[:, :k] / d).astype(np.int64) if scheme == 'cubic' else snap_bcc(z[:, :3], d)[1]
				cells[key].append(np.unique(ids, axis=0))
				if len(cells[key]) > 4: cells[key] = [np.unique(np.concatenate(cells[key], 0), axis=0)]
			if done % (10 * CHUNK) == 0: print('sampled', done, time.strftime('%X'), flush=True)
		for key in keys: cells[key] = [np.unique(np.concatenate(cells[key], 0), axis=0)]
		counts[target] = {key: len(cells[key][0]) for key in keys}
		print('N samples', target, {str(key): v for key, v in counts[target].items()}, flush=True)

	rows = []
	for key in keys:
		scheme, k, d = key
		snap = np.rint(ci[:, :k] / d) * d if scheme == 'cubic' else snap_bcc(ci[:, :3], d)[0]
		row = dict(scheme=scheme, k=k, spacing=d, occ={int(n): counts[n][key] for n in NS}, **coverage(snap))
		rows.append(row); print(json.dumps(row), flush=True)
	json.dump(dict(regressor_rms=[float(x) for x in rms], rows=rows), open(os.path.join(F.OUT, 'svd_refined_ecc5d.json'), 'w'), indent=1)

if __name__ == '__main__':
	main()
