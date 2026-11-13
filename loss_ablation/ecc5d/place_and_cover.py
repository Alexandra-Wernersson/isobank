"""
Downstream test of one arm:  python place_and_cover.py <arm> <seed>  (or <array index>)

1. Greedy placement with THE ARM'S flow at radius_factor * r_target
   (same settings and placement seed for every arm).
2. Coverage with TRUE matches on fixed injections (seed 4002, uniform in the
   coordinates, m1 >= m2). Candidates are shortlisted with the fixed REFERENCE
   flow (k nearest), identical for every arm, so a poor arm flow cannot fake
   misses through bad retrieval; every reported match is exact.

Writes runs/<arm>_s<seed>/{bank.npy, ff.npy, downstream.json}.
"""
import sys, json, time
import numpy as np, torch
from scipy.spatial import cKDTree
from common import *
from isobank import load_checkpoint, greedy_placement

#	STOP_ACC = 0.01 as in place_radius_scan.py (the paper-era 1.45 bank, 2.18M).
#	The first pass used the config's 0.10, which stops before the low-mass
#	corner fills (~1M templates, 87% missed at Mc < 1.1); those files are kept
#	without the '_st001' suffix.
RADIUS_FACTOR, STOP_ACC, STOP_ON, PLACE_SEED = 1.45, 0.01, 'coord', 0
N_INJ, K, INJ_SEED, THR, CHUNK = 4000, 200, 4002, 0.97, 20_000
MC_EDGES = [0.87, 1.10, 1.40, 1.80, 2.40, 3.20, 4.0]

def to_latent(flow, a):
	out = []
	for i in range(0, len(a), 200_000):
		with torch.no_grad():
			out.append(flow(torch.tensor(a[i:i+200_000], dtype=torch.float32))[0].numpy())
	return np.vstack(out)

def main():
	pos = [a for a in sys.argv[1:] if '=' not in a]; kv = parse_kv(sys.argv[1:])
	p, which = int(kv.get('p', 0)), kv.get('ckpt', 'best')
	if len(pos) == 1:
		i = int(pos[0]); arm, seed = ARM_ORDER[i % len(ARMS)], SEEDS[i // len(ARMS)]
	else:
		arm, seed = pos[0], int(pos[1])
	if arm == 'shipped':                              # reference: the repo's shipped flow
		d = os.path.join(RUNS, 'shipped'); os.makedirs(d, exist_ok=True)
	else:
		d = run_dir(arm, seed, p)
	sfx = ('' if which == 'best' else f'_{which}') + '_st001'
	C = load_config(); metric = C.make_metric()
	flow, ck = load_checkpoint(REF_FLOW if arm == 'shipped' else os.path.join(d, f'flow_{which}.pt'))
	r_target = float(ck['r_target'])
	print(f'=== {arm} s{seed} p{p} [{which}]: placement at {RADIUS_FACTOR} r_target ==='); sys.stdout.flush()

	bank_path = os.path.join(d, f'bank{sfx}.npy')
	t0 = time.time()
	if not os.path.exists(bank_path):
		partial = os.path.join(d, f'bank{sfx}_partial.npy')
		init = np.load(partial) if os.path.exists(partial) else None
		bank = greedy_placement(flow, RADIUS_FACTOR * r_target, C.lo, C.hi,
		                        embed_fn=C.embed_fn, valid_fn=C.valid_mask,
		                        batch=50_000, max_batches=5_000,
		                        stop_acceptance=STOP_ACC, stop_on=STOP_ON,
		                        seed=PLACE_SEED, initial_templates=init,
		                        partial_save=partial, verbose=True)
		np.save(bank_path, bank)
	bank = np.load(bank_path).astype(np.float32)
	t_place = (time.time() - t0) / 60
	print(f'bank: {len(bank):,} templates [{t_place:.1f} min]'); sys.stdout.flush()

	#	injections (same draw as scripts/evaluate_bank.py with --seed 4002)
	rng = np.random.default_rng(INJ_SEED); draws, have = [], 0
	while have < N_INJ:
		b = rng.uniform(C.lo, C.hi, (4 * N_INJ, C.D)).astype(np.float32)
		b = b[C.valid_mask(b)]; draws.append(b); have += len(b)
	sig = np.vstack(draws)[:N_INJ]

	ref, _ = load_checkpoint(REF_FLOW)
	k = min(K, len(bank))
	dist, idx = cKDTree(to_latent(ref, bank)).query(to_latent(ref, sig), k=k, workers=-1)
	t1 = np.repeat(sig, k, axis=0); t2 = bank[np.asarray(idx).reshape(-1)]
	m = np.empty(len(t1)); t0 = time.time()
	for s in range(0, len(t1), CHUNK):
		e = min(s + CHUNK, len(t1))
		with torch.no_grad():
			m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e]), torch.tensor(t2[s:e])))
		if (s // CHUNK) % 25 == 0:
			print(f'  {100*e/len(t1):5.1f}%  [{(time.time()-t0)/60:.0f} min]'); sys.stdout.flush()
	m = np.nan_to_num(m.reshape(len(sig), k), nan=-np.inf)
	running = np.maximum.accumulate(m, axis=1); ff = running[:, -1]
	np.save(os.path.join(d, f'ff{sfx}.npy'), ff)

	miss = (ff < THR)
	m1, m2 = sig[:, 0].astype(float), sig[:, 1].astype(float)
	mc = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
	res = dict(arm=arm, seed=seed, pair_seed=p, ckpt=which, stop_acceptance=STOP_ACC, epoch=ck.get('epoch'), n_templates=int(len(bank)), place_minutes=t_place,
	           n_inj=int(N_INJ), k=int(k), radius_factor=RADIUS_FACTOR,
	           miss=float(miss.mean()), miss_se=float(np.sqrt(miss.mean()*(1-miss.mean())/N_INJ)),
	           ff_p1=float(np.percentile(ff, 1)), ff_min=float(ff.min()),
	           miss_vs_k={int(kk): float((running[:, kk-1] < THR).mean())
	                      for kk in [1, 20, 50, 100, 200, 500] if kk <= k},
	           miss_high_e=float(miss[sig[:, C.E_COL] >= 0.12].mean()),
	           miss_by_mc=[dict(lo=a, hi=b, n=int(((mc >= a) & (mc < b)).sum()),
	                            miss=float(miss[(mc >= a) & (mc < b)].mean()))
	                       for a, b in zip(MC_EDGES[:-1], MC_EDGES[1:]) if ((mc >= a) & (mc < b)).sum() > 0])
	with open(os.path.join(d, f'downstream{sfx}.json'), 'w') as f:
		json.dump(res, f, indent=1)
	print(json.dumps(res, indent=1))

if __name__ == '__main__':
	main()
