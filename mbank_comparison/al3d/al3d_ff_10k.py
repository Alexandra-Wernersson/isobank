"""
al3d_ff_10k.py — aligned-spin 3D FF at N_inj = 10,000 (FF table + FF CDF figure).

Same draw as al3d_kconv.py / ff_mbank_al3d_k500.py (coordinate-uniform, seed 4242),
extended to 10,000: default_rng fills row by row, so the first 2,000 injections
are exactly the earlier ones.

k=100: the k=2000 convergence run was flat from k=20 on (best-match rank max 52),
so k=100 is converged; the k-table below re-checks it on these injections.

Banks: isobank shipped (templates.npy, 23,810, placed at r_z), isobank r_target
(templates_ab_rtarget.npy, 25,552), mbank stochastic (23,167).
"""
import os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..'))
os.chdir(HERE)
import config as C
from isobank.flow import load_checkpoint

K, THR, CHUNK, N_INJ = 100, 0.97, 20_000, 10_000
MB = '/gpfs/home5/awernersson/projects/mbank/'
flow, ckpt = load_checkpoint('flow_best.pt'); metric = C.make_metric()
r_target = float(ckpt['r_target'])

def fz(a):
    o = []
    for i in range(0, len(a), 200_000):
        with torch.no_grad(): z, _ = flow(torch.tensor(a[i:i+200_000], dtype=torch.float32))
        o.append(z.numpy())
    return np.vstack(o)

sig = np.random.default_rng(4242).uniform(C.lo, C.hi, (N_INJ, C.D)).astype(np.float32)
z_sig = fz(sig)

BANKS = [('isobank_shipped',  'templates.npy'),
         ('isobank_rtarget',  'templates_ab_rtarget.npy'),
         ('mbank_stochastic', MB + 'bank_al3d_mbank_stochastic.npy')]
for lab, path in BANKS:
    bank = np.load(path).astype(np.float32)
    dist, idx = cKDTree(fz(bank)).query(z_sig, k=K, workers=-1)
    reach = dist[:, K-1] / r_target
    print(f"\n{lab}: {len(bank):,} templates   shortlist reach / r_target: min {reach.min():.2f}  "
          f"median {np.median(reach):.2f}"); sys.stdout.flush()
    t1 = np.repeat(sig, K, axis=0); t2 = bank[idx.reshape(-1)]
    m = np.empty(len(t1)); t0 = time.time()
    for s in range(0, len(t1), CHUNK):
        e = min(s+CHUNK, len(t1))
        with torch.no_grad():
            m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e], dtype=torch.float32),
                                             torch.tensor(t2[s:e], dtype=torch.float32)))
    m = np.nan_to_num(m.reshape(N_INJ, K), nan=-np.inf)
    run = np.maximum.accumulate(m, axis=1)
    np.save(f'ff_al3d_{lab}_k{K}_N10k_seed4242.npy', run[:, K-1])
    print(f"  [{(time.time()-t0)/60:.0f} min]\n{'k':>6} | {'miss [%]':>9}")
    for kk in [1, 5, 20, 50, 100]:
        print(f"{kk:>6} | {100*(run[:, kk-1] < THR).mean():9.2f}")
    p = (run[:, K-1] < THR).mean()
    print(f"  FINAL {lab}: miss {100*p:.2f} +/- {100*np.sqrt(p*(1-p)/N_INJ):.2f} %   "
          f"(first 2,000 only: {100*(run[:2000, K-1] < THR).mean():.2f} %)")
    sys.stdout.flush()
