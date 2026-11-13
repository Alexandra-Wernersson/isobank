"""
Does L_glob matter in the aligned-spin 3D space?  (Companion to eccentric_5d_loss_ablation.)

  python al3d_ablation.py pairs            cache one training-pair set (no edge term in 3D)
  python al3d_ablation.py train <i>        i = 0: full s0, 1: no_global s0, 2: no_global s1
  python al3d_ablation.py score <i>        0-2 as above, 3: the paper's trained flow (full recipe)

Release isobank package and the paper's 3D config (M 30-50, q 1-5, chi, IMRPhenomD).
Scoring: FINAL checkpoint, radius searched for the paper's 23,810 templates (+/-3%), stop 0.01,
placement seed 0, 5000 injections (seed 4242, the first 5000 of the paper's 10,000), k=20
(converged in this space) shortlisted with the paper's flow, true matches, threshold 0.97.
"""
import os, sys, json, importlib.util
import numpy as np, torch
ISO = os.environ.get('ISOBANK_ROOT', '/home/awernersson/projects/isobank'); sys.path.insert(0, ISO)
from isobank import (long_range_pairs, metric_guided_pairs, filter_by_distance, train_isometric_flow,
                     load_checkpoint, greedy_placement)
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs')
PAPER = os.environ.get('AL3D_PAPER_DIR', '/home/awernersson/projects/isobank_testing/examples/aligned_3d')   # paper's 3D config.py, flow_best.pt, r_target.npy
spec = importlib.util.spec_from_file_location('c3', os.path.join(PAPER, 'config.py'))
C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
PAIRS = os.path.join(RUNS, 'train_pairs.pt')
ARMS = [('full', (1., 5., 10., 0.), 0), ('no_global', (0., 5., 10., 0.), 0), ('no_global', (0., 5., 10., 0.), 1),
        None,    # index 3 is the paper's flow in score(), never trained here
        ('full', (1., 5., 10., 0.), 1), ('full', (1., 5., 10., 0.), 2), ('full', (1., 5., 10., 0.), 3),    # 4-6
        ('no_global', (0., 5., 10., 0.), 2), ('no_global', (0., 5., 10., 0.), 3)]    # 7-8
TARGET, TOL, MAXIT, F0, SLOPE0 = 23810, 0.03, 6, 1.0, -2.5
N_INJ, K, INJ_SEED, THR, CHUNK = 5000, 20, 4242, 0.97, 20000

def pairs():
	metric = C.make_metric(); r = float(np.load(C.R_TARGET_FILE)[0])
	tr, va = C.sample_data()
	lp = long_range_pairs(tr, metric, C.N_LONG, d_max=C.LONG_DMAX, seed=42, phi_col=None)
	gt = metric.get_metric(torch.tensor(tr), metric_type='hessian'); gv = metric.get_metric(torch.tensor(va), metric_type='hessian')
	t1, t2 = metric_guided_pairs(tr, gt, r, C.N_LOCAL, C.LOCAL_FRACS, C.lo, C.hi, seed=123)
	loc = filter_by_distance(metric, t1, t2, C.LOCAL_DMIN_R * r, np.inf)
	t1, t2 = metric_guided_pairs(tr, gt, r, C.N_BOUND, C.BOUND_FRACS, C.lo, C.hi, seed=555)
	bnd = filter_by_distance(metric, t1, t2, .5 * r, 1.5 * r)
	t1, t2 = metric_guided_pairs(va, gv, r, C.N_MON, C.BOUND_FRACS, C.lo, C.hi, seed=999)
	mon = filter_by_distance(metric, t1, t2, .5 * r, 1.5 * r)
	torch.save(dict(long=lp, local=loc, bound=bnd, monitor=mon, train_data=tr, r_target=r), PAIRS)
	print('pairs', {k: len(v[2]) for k, v in dict(long=lp, local=loc, bound=bnd, monitor=mon).items()})

def train(i):
	arm, w, seed = ARMS[i]; d = os.path.join(RUNS, f'{arm}_s{seed}'); os.makedirs(d, exist_ok=True)
	torch.manual_seed(seed); np.random.seed(seed)
	metric = C.make_metric(); P = torch.load(PAIRS, weights_only=False)
	train_isometric_flow(metric, P['r_target'], C.variable_format, P['long'], P['local'], P['bound'], P['monitor'],
	    d_flow=C.d_flow, n_layers=C.n_layers, hidden_dim=C.hidden_dim, masks=C.masks, depth=C.DEPTH, weights=w,
	    tail_w=C.TAIL_W, embed_fn=None, train_data=P['train_data'], phi_col=None, n_epochs=C.N_EPOCHS,
	    best_path=os.path.join(d, 'flow_best.pt'), final_path=os.path.join(d, 'flow_final.pt'), extra_ckpt=dict(arm=arm, seed=seed))

def to_z(f, a):
	o = []
	for s in range(0, len(a), 200000):
		with torch.no_grad(): o.append(f(torch.tensor(a[s:s + 200000], dtype=torch.float32))[0].numpy())
	return np.vstack(o)

def score(i):
	if i == 3: name, path = 'paper_full', os.path.join(PAPER, 'flow_final.pt')
	else: arm, _, seed = ARMS[i]; name, path = f'{arm}_s{seed}', os.path.join(RUNS, f'{arm}_s{seed}', 'flow_final.pt')
	out = os.path.join(RUNS, name); os.makedirs(out, exist_ok=True)
	flow, ck = load_checkpoint(path); r = float(ck['r_target']); metric = C.make_metric(); print(f'=== {name} ===', flush=True)
	hist, best, f = [], None, F0
	for it in range(MAXIT):
		b = greedy_placement(flow, f * r, C.lo, C.hi, valid_fn=None, stop_acceptance=C.STOP_ACCEPTANCE, stop_on='coord', seed=0, verbose=False)
		n = len(b); hist.append(dict(f=f, N=n)); print(f'  radius {f:.3f}: N={n:,}', flush=True)
		if best is None or abs(np.log(n / TARGET)) < abs(np.log(len(best[1]) / TARGET)): best = (f, np.asarray(b).astype(np.float32))
		if abs(n / TARGET - 1) <= TOL: break
		sl = SLOPE0
		if len(hist) > 1 and hist[-1]['N'] != hist[-2]['N']: sl = float(np.clip(np.log(hist[-1]['N'] / hist[-2]['N']) / np.log(hist[-1]['f'] / hist[-2]['f']), -6, -1))
		f = float(np.clip(f * (TARGET / n) ** (1 / sl), .4, 4))
	f, bank = best; np.save(os.path.join(out, 'bank.npy'), bank)
	sig = np.random.default_rng(INJ_SEED).uniform(C.lo, C.hi, (N_INJ, C.D)).astype(np.float32)
	ref, _ = load_checkpoint(os.path.join(PAPER, 'flow_best.pt')); k = min(K, len(bank))
	_, idx = cKDTree(to_z(ref, bank)).query(to_z(ref, sig), k=k, workers=-1)
	t1 = np.repeat(sig, k, axis=0); t2 = bank[np.asarray(idx).reshape(-1)]; m = np.empty(len(t1))
	for s in range(0, len(t1), CHUNK):
		e = min(s + CHUNK, len(t1))
		with torch.no_grad(): m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e]), torch.tensor(t2[s:e])))
	m = np.nan_to_num(m.reshape(N_INJ, k), nan=-np.inf); run = np.maximum.accumulate(m, axis=1); ff = run[:, -1]
	np.save(os.path.join(out, 'ff.npy'), ff); miss = ff < THR
	res = dict(name=name, n_templates=len(bank), radius_factor=f, search=hist, n_inj=N_INJ, k=k, miss=float(miss.mean()),
	           miss_se=float(np.sqrt(miss.mean() * (1 - miss.mean()) / N_INJ)), ff_min=float(ff.min()),
	           ff_p=[float(x) for x in np.percentile(ff, [.1, 1, 5, 50])], miss_k1=float((run[:, 0] < THR).mean()), miss_k10=float((run[:, 9] < THR).mean()))
	json.dump(res, open(os.path.join(out, 'matched.json'), 'w'), indent=1); print(json.dumps({x: res[x] for x in ('n_templates', 'radius_factor', 'miss', 'miss_se', 'miss_k10', 'ff_p', 'ff_min')}))

if __name__ == '__main__':
	{'pairs': lambda: pairs(), 'train': lambda: train(int(sys.argv[2])), 'score': lambda: score(int(sys.argv[2]))}[sys.argv[1]]()
