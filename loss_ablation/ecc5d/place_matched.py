"""
Matched-size ablation:  python place_matched.py <index>

Each run is compared at the SAME bank size. For the run's FINAL checkpoint, search the placement
radius (secant on ln N vs ln radius) until the greedy bank has TARGET templates to within TOL,
then score it with true matches. Everything else matches place_and_cover.py: stop 0.01 (coord),
placement seed 0, injection seed 4002, shortlist from the fixed shipped flow.

Index order: 7 arms x seeds 0,1 (0-13), then the three resampled-pair full runs (14-16).
Writes <run>/matched_final.json, bank_matched_final.npy, ff_matched_final.npy.
"""
import sys, json, time
import numpy as np, torch
from scipy.spatial import cKDTree
from common import *
from isobank import load_checkpoint, greedy_placement
from place_and_cover import to_latent, MC_EDGES, STOP_ACC, STOP_ON, PLACE_SEED, THR, CHUNK, INJ_SEED

E = os.environ.get
TARGET = int(E('ABL_TARGET', 2_267_558)); TOL = float(E('ABL_TOL', 0.03)); MAX_ITER = int(E('ABL_MAXIT', 5))
F0 = float(E('ABL_F0', 1.45)); N_INJ = int(E('ABL_NINJ', 4000)); K = int(E('ABL_K', 200))
OUT_OVERRIDE = E('ABL_OUT')
RUN_LIST = [(a, s, 0) for s in SEEDS for a in ARM_ORDER] + [('full', 100 + p, p) for p in VAR_PAIR_SEEDS] + [('loc_edge', s, 0) for s in SEEDS] \
	+ [('no_bound', s, 0) for s in (2, 3)] + [('no_tail', s, 0) for s in (2, 3)]   # 17-18 loc_edge, 19-20 no_bound, 21-22 no_tail
RUN_LIST += [(a, s, 0) for a in WEIGHT_ARMS for s in SEEDS]   # 23-38 weight sensitivity, 8 settings x seeds 0,1
SLOPE0, F_MIN, F_MAX = -3.5, 0.5, 6.0        # d ln N / d ln radius, from the 1.6 -> 1.45 r_target scan

def main():
	arm, seed, p = RUN_LIST[int(sys.argv[1])]
	src = run_dir(arm, seed, p); out = OUT_OVERRIDE or src
	C = load_config(); metric = C.make_metric()
	flow, ck = load_checkpoint(os.path.join(src, 'flow_final.pt'))
	r_target = float(ck['r_target'])
	print(f'=== {run_name(arm, seed, p)} [final]: search radius for N = {TARGET:,} ==='); sys.stdout.flush()

	hist, best, f = [], None, F0
	for it in range(MAX_ITER):
		t0 = time.time()
		bank = greedy_placement(flow, f * r_target, C.lo, C.hi, embed_fn=C.embed_fn, valid_fn=C.valid_mask,
		                        batch=50_000, max_batches=5_000, stop_acceptance=STOP_ACC, stop_on=STOP_ON,
		                        seed=PLACE_SEED, verbose=False)
		N = len(bank); hist.append(dict(f=f, N=N, minutes=(time.time() - t0) / 60))
		print(f'  iter {it}: radius {f:.4f} r_target -> N = {N:,}  [{hist[-1]["minutes"]:.1f} min]'); sys.stdout.flush()
		if best is None or abs(np.log(N / TARGET)) < abs(np.log(len(best[1]) / TARGET)):
			best = (f, bank)
		if abs(N / TARGET - 1) <= TOL: break
		slope = SLOPE0
		if len(hist) > 1 and hist[-1]['N'] != hist[-2]['N'] and hist[-1]['f'] != hist[-2]['f']:
			slope = float(np.clip(np.log(hist[-1]['N'] / hist[-2]['N']) / np.log(hist[-1]['f'] / hist[-2]['f']), -8.0, -1.0))
		f = float(np.clip(f * (TARGET / N) ** (1 / slope), F_MIN, F_MAX))
	f, bank = best
	bank = np.asarray(bank).astype(np.float32)
	np.save(os.path.join(out, 'bank_matched_final.npy'), bank)
	print(f'matched bank: {len(bank):,} templates at {f:.4f} r_target'); sys.stdout.flush()

	rng = np.random.default_rng(INJ_SEED); draws, have = [], 0
	while have < N_INJ:
		b = rng.uniform(C.lo, C.hi, (4 * N_INJ, C.D)).astype(np.float32)
		b = b[C.valid_mask(b)]; draws.append(b); have += len(b)
	sig = np.vstack(draws)[:N_INJ]
	ref, _ = load_checkpoint(REF_FLOW); k = min(K, len(bank))
	_, idx = cKDTree(to_latent(ref, bank)).query(to_latent(ref, sig), k=k, workers=-1)
	idx = np.asarray(idx).reshape(len(sig), k)
	t1 = np.repeat(sig, k, axis=0); t2 = bank[idx.reshape(-1)]; m = np.empty(len(t1))
	for s in range(0, len(t1), CHUNK):
		e = min(s + CHUNK, len(t1))
		with torch.no_grad():
			m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e]), torch.tensor(t2[s:e])))
	m = np.nan_to_num(m.reshape(len(sig), k), nan=-np.inf)
	running = np.maximum.accumulate(m, axis=1); ff = running[:, -1]
	np.save(os.path.join(out, 'ff_matched_final.npy'), ff)
	miss = ff < THR
	m1, m2 = sig[:, 0].astype(float), sig[:, 1].astype(float)
	mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
	res = dict(arm=arm, seed=seed, pair_seed=p, ckpt='final', target=TARGET, radius_factor=f, search=hist,
	           n_templates=int(len(bank)), n_inj=N_INJ, k=int(k), stop_acceptance=STOP_ACC,
	           miss=float(miss.mean()), miss_se=float(np.sqrt(miss.mean() * (1 - miss.mean()) / N_INJ)),
	           ff_p1=float(np.percentile(ff, 1)), ff_min=float(ff.min()),
	           miss_vs_k={int(kk): float((running[:, kk - 1] < THR).mean()) for kk in [1, 20, 50, 100, 200] if kk <= k},
	           miss_high_e=float(miss[sig[:, C.E_COL] >= 0.12].mean()),
	           miss_by_mc=[dict(lo=a, hi=b, n=int(((mc >= a) & (mc < b)).sum()), miss=float(miss[(mc >= a) & (mc < b)].mean()))
	                       for a, b in zip(MC_EDGES[:-1], MC_EDGES[1:]) if ((mc >= a) & (mc < b)).sum() > 0])
	json.dump(res, open(os.path.join(out, 'matched_final.json'), 'w'), indent=1)
	print(json.dumps({k_: res[k_] for k_ in ('n_templates', 'radius_factor', 'miss', 'miss_se', 'miss_high_e', 'ff_min')}, indent=1))

if __name__ == '__main__':
	main()
