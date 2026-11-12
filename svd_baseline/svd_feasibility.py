"""
Feasibility of the Cabass et al. (arXiv:2511.10466) waveform representation on the isobank spaces.

Their autoencoder takes the top 10 SVD coefficients of the PSD-weighted waveform phase, with one
fixed amplitude profile per bank. This measures, on held-out waveforms:
  * how much the amplitude profile varies across the space (their pipeline assumes it does not), and
  * the overlap between each waveform and its rank-k SVD phase reconstruction, overall and by Mc.

Usage: python svd_feasibility.py <ecc5d|al3d>
"""
import sys, json, os
import numpy as np, lal, lalsimulation as ls
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ISO = os.path.join(HERE, 'psds')
OUT = os.path.join(HERE, 'results'); os.makedirs(OUT, exist_ok=True)
SPACES = {
	'ecc5d': dict(approx=ls.TaylorF2Ecc, fmin=15., fmax=1024., df_fine=1/1024., psd=f'{ISO}/aligo_O3actual_H1.txt', asd=True),
	'al3d':  dict(approx=ls.IMRPhenomD,  fmin=10., fmax=1024., df_fine=1/64.,   psd=f'{ISO}/aLIGOZeroDetHighPower.txt', asd=False),
}
SPACE = sys.argv[1]; S = SPACES[SPACE]
N_TRAIN, N_TEST, KS = 2500, 800, [2, 3, 5, 10, 20, 50, 100, 200]
GRID = np.arange(S['fmin'], S['fmax'], 0.25)

def draw(n, seed):
	rng = np.random.default_rng(seed)
	if SPACE == 'ecc5d':
		x = rng.uniform([1, 1, -.9, -.05, 1e-5], [7, 3, .9, .05, .15], (4 * n, 5))
		x = x[x[:, 0] >= x[:, 1]][:n]
		return np.c_[x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]]
	x = rng.uniform([30, 1, -.99], [50, 5, .99], (n, 3))
	m1 = x[:, 0] * x[:, 1] / (1 + x[:, 1]); m2 = x[:, 0] / (1 + x[:, 1])
	return np.c_[m1, m2, x[:, 2], x[:, 2], 0 * m1]

def waveform(p):
	m1, m2, s1, s2, e = map(float, p)
	hp, _ = ls.SimInspiralChooseFDWaveform(m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0, 0, s1, 0, 0, s2, 1e6 * lal.PC_SI, 0., 0., 0.,
	                                       e, 0., S['df_fine'], S['fmin'], S['fmax'], S['fmin'], lal.CreateDict(), S['approx'])
	h = hp.data.data; f = np.arange(len(h)) * S['df_fine']
	i0 = int(np.ceil(S['fmin'] / S['df_fine']))
	amp = np.abs(h[i0:]); ok = amp > 0
	last = np.nonzero(ok)[0][-1]
	psi = np.unwrap(np.angle(h[i0:last + 1])); f_ok = f[i0:last + 1]
	A = np.interp(GRID, f[i0:], amp, right=0.)
	P = np.interp(GRID, f_ok, psi)
	over = GRID > f_ok[-1]
	if over.any():                       # continue the phase linearly beyond the waveform's cutoff
		slope = (psi[-1] - psi[-1 - 256]) / (f_ok[-1] - f_ok[-1 - 256])
		P[over] = psi[-1] + slope * (GRID[over] - f_ok[-1])
	return A.astype(np.float64), P

def main():
	f_psd, s = np.loadtxt(S['psd']).T[:2]
	psd = np.interp(GRID, f_psd, s ** 2 if S['asd'] else s)
	X = draw(N_TRAIN + N_TEST, 11)
	with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 8))) as pool:
		out = pool.map(waveform, X, chunksize=16)
	A = np.array([o[0] for o in out]); P = np.array([o[1] for o in out])
	mc = (X[:, 0] * X[:, 1]) ** .6 / (X[:, 0] + X[:, 1]) ** .2

	W = A / np.sqrt(psd); W /= np.sqrt((W ** 2).sum(1, keepdims=True))      # per-waveform normalised weights
	w = W[:N_TRAIN].mean(0); w /= np.sqrt((w ** 2).sum())                    # one amplitude profile for the bank
	amp_ov = (W[N_TRAIN:] * w).sum(1)

	#	remove the mean phase, and a + b f per waveform (time and phase maximisation)
	B = np.c_[np.ones_like(GRID), GRID - GRID.mean()] * w[:, None]
	Q, _ = np.linalg.qr(B)
	mean = P[:N_TRAIN].mean(0)
	for _ in range(3):
		R = (P - mean) * w; R -= (R @ Q) @ Q.T
		mean = mean + (R[:N_TRAIN].mean(0) / np.where(w > 1e-12, w, np.inf))
	R = (P - mean) * w; R -= (R @ Q) @ Q.T
	_, sv, Vt = np.linalg.svd(R[:N_TRAIN], full_matrices=False)
	test = R[N_TRAIN:]; mc_t = mc[N_TRAIN:]
	bins = [(0, 1.4), (1.4, 2.4), (2.4, 99)] if SPACE == 'ecc5d' else [(0, 12), (12, 16), (16, 99)]

	res = dict(space=SPACE, n_train=N_TRAIN, n_test=N_TEST, amp_overlap_p=[float(x) for x in np.percentile(amp_ov, [0, 1, 5, 50])],
	           n_grid=len(GRID), sv_top=[float(x) for x in sv[:12]], residual_rad_rms=float(np.sqrt((test ** 2).sum(1).mean())), k={})
	mask = w > 1e-12
	for k in KS:
		V = Vt[:k]; rec = (test @ V.T) @ V
		d = np.zeros_like(rec); d[:, mask] = (rec - test)[:, mask] / w[mask]
		ov = (w ** 2 * np.cos(d)).sum(1)
		row = dict(frac_ge_097=float((ov >= .97).mean()), frac_ge_099=float((ov >= .99).mean()), p1=float(np.percentile(ov, 1)), median=float(np.median(ov)))
		row['by_mc'] = {f'{a}-{b}': dict(n=int(((mc_t >= a) & (mc_t < b)).sum()), frac_ge_097=float((ov[(mc_t >= a) & (mc_t < b)] >= .97).mean()),
		                                 median=float(np.median(ov[(mc_t >= a) & (mc_t < b)]))) for a, b in bins}
		res['k'][k] = row; print(k, json.dumps(row), flush=True)
	print('amp overlap p0,1,5,50', res['amp_overlap_p'], ' top sv', res['sv_top'][:6], ' residual rms norm', res['residual_rad_rms'])
	json.dump(res, open(os.path.join(OUT, f'svd_feasibility_{SPACE}.json'), 'w'), indent=1)

if __name__ == '__main__':
	main()
