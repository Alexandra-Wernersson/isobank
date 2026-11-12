"""
Number of occupied cells of a cubic SVD grid, and its coverage, from svd_coeffs_<space>.npz.

A small MLP maps parameters -> SVD coefficients (trained on the saved waveforms, checked on the injections).
It is sampled densely, with half the samples log-uniform in the masses because uniform draws starve the low-mass
corner where most templates are, and the distinct grid cells it reaches are counted: that is the bank size.

Coverage uses the saved injection coefficients: FF ~ 1 - (|snap error|^2 + |truncated coefficients|^2) / 2 (the small-angle form of
the phase overlap; phase only, one amplitude profile per bank, no extra time/phase maximisation).

Usage: python svd_grid_occupancy.py <ecc5d|al3d>
"""
import sys, os, json, time
import numpy as np, torch, torch.nn as nn

SPACE = sys.argv[1]; HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results'); Z = np.load(os.path.join(OUT, f'svd_coeffs_{SPACE}.npz')); C, X, n1, n2 = Z['C'], Z['X'], int(Z['n1']), int(Z['n2'])
KS, DELTAS, KMAX = [3, 4, 5], [.15, .2, .25, .283, .32, .36, .4, .5], 5
NS = [10_000_000, 30_000_000, 100_000_000]; CHUNK = 2_000_000
torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', 8)))

def feats(p):
	m1, m2, s1, s2, e = [torch.as_tensor(p[:, i], dtype=torch.float64) for i in range(5)]
	mc = (m1 * m2) ** .6 / (m1 + m2) ** .2
	return torch.stack([torch.log(m1), torch.log(m2), s1, s2, e, mc ** (-5 / 3) / 1, mc ** -1, m2 / m1], 1)

def sample(n, rng):
	if SPACE == 'ecc5d':
		lo, hi = np.array([1, 1, -.9, -.05, 1e-5]), np.array([7, 3, .9, .05, .15])
		x = rng.uniform(lo, hi, (n, 5))
		h = n // 2; x[:h, 0] = np.exp(rng.uniform(np.log(1), np.log(7), h)); x[:h, 1] = np.exp(rng.uniform(0, np.log(3), h))
		bad = x[:, 0] < x[:, 1]; x[bad, 0], x[bad, 1] = x[bad, 1].copy(), x[bad, 0].copy()
		return x
	M, q, chi = rng.uniform(30, 50, n), rng.uniform(1, 5, n), rng.uniform(-.99, .99, n)
	return np.c_[M * q / (1 + q), M / (1 + q), chi, chi, 0 * M]

Xt, Ct = feats(X[:n1 + n2]), torch.tensor(C[:n1 + n2, :KMAX], dtype=torch.float64)
mx, sx, my, sy = Xt.mean(0), Xt.std(0), Ct.mean(0), Ct.std(0)
sx[sx == 0] = 1
net = nn.Sequential(nn.Linear(8, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, KMAX)).double()
opt = torch.optim.Adam(net.parameters(), 2e-3); N_EP = 20000; sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, N_EP)
xn, yn = (Xt - mx) / sx, (Ct - my) / sy
for ep in range(N_EP):
	opt.zero_grad(); l = ((net(xn) - yn) ** 2).mean(); l.backward(); opt.step(); sch.step()
	if ep % 4000 == 0: print(ep, float(l), flush=True)

def predict(p):
	with torch.no_grad(): return (net((feats(p) - mx) / sx) * sy + my).numpy()

ci = C[n1 + n2:]; err = predict(X[n1 + n2:]) - ci[:, :KMAX]
print('regressor error on injections (rms per coefficient):', np.round(np.sqrt((err ** 2).mean(0)), 4), 'max', np.round(np.abs(err).max(0), 3), flush=True)

rows = []; rng = np.random.default_rng(5)
cells = {(k, d): [] for k in KS for d in DELTAS}; done = 0; counts = {}
for target in NS:
	while done < target:
		z = predict(sample(CHUNK, rng)); done += CHUNK
		for (k, d), st in cells.items():
			ik = np.rint(z[:, :k] / d).astype(np.int64)
			st.append(np.unique(ik, axis=0))
		if done % (10 * CHUNK) == 0: print('sampled', done, time.strftime('%X'), flush=True)
		for key_, st in cells.items():      # keep memory bounded
			if len(st) > 4:
				cells[key_] = [np.unique(np.concatenate(st, 0), axis=0)]
	for key_, st in cells.items():
		cells[key_] = [np.unique(np.concatenate(st, 0), axis=0)]
	counts[target] = {str(key_): len(st[0]) for key_, st in cells.items()}
	print('N samples', target, flush=True)

for k in KS:
	for d in DELTAS:
		snap = np.rint(ci[:, :k] / d) * d
		ov = 1 - (((ci[:, :k] - snap) ** 2).sum(1) + (ci[:, k:] ** 2).sum(1)) / 2
		row = dict(k=k, delta=d, cov=float((ov >= .97).mean()), p1=float(np.percentile(ov, 1)), worst=float(ov.min()), occ={int(n): counts[n][str((k, d))] for n in NS})
		rows.append(row); print(json.dumps(row), flush=True)
json.dump(dict(space=SPACE, regressor_rms=[float(x) for x in np.sqrt((err ** 2).mean(0))], rows=rows), open(os.path.join(OUT, f'svd_occupancy_{SPACE}.json'), 'w'), indent=1)
