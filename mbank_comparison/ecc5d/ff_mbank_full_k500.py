"""
ff_mbank_full_k500.py -- mbank's FULL native eccentric bank (2,953,739 templates,
no truncation) scored on the same 10,000 injections (seed 4002) and k=500 shortlist
as the isobank ALL cell and the matched-size mbank run.

Purpose: if mbank's entire bank still covers less than isobank's 2,267,558-template
bank (96.64%), the paper can state a MEASURED template reduction
(1 - 2,267,558/2,953,739 = 23.2%) instead of the old k=20 extrapolation.

The shortlist is only a search index in isobank's latent space; the fitting factor is
the true TaylorF2Ecc match. Completeness at the 0.97 threshold is re-checked here for
this larger (denser) bank: no pair reaches match >= 0.97 beyond 1.85 r_target, so the
k=500 shortlist radius must exceed that for every injection.
"""
import os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/gpfs/home5/awernersson/projects/isobank_testing')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import config as C
from isobank.flow import load_checkpoint

K, THR, CHUNK, N_INJ, ENVELOPE = 500, 0.97, 20_000, 10_000, 1.85
MBANK = '/gpfs/home5/awernersson/projects/mbank/ecc5D_mbank_native/bank_ecc5D_native.dat'

flow, ckpt = load_checkpoint(C.BEST_CKPT); metric = C.make_metric()
r_target = float(ckpt['r_target'])
def fz(a):
    o = []
    for i in range(0, len(a), 200_000):
        with torch.no_grad(): z, _ = flow(torch.tensor(a[i:i+200_000], dtype=torch.float32))
        o.append(z.numpy())
    return np.vstack(o)
def draw_ecc(n, seed):
    rng = np.random.default_rng(seed); o, h = [], 0
    while h < n:
        b = rng.uniform(C.lo, C.hi, (4*n, C.D)).astype(np.float32)
        b = b[b[:, 0] >= b[:, 1]]; o.append(b); h += len(b)
    return np.vstack(o)[:n]

sig = draw_ecc(N_INJ, 4002)
bank = np.loadtxt(MBANK)[:, :C.D].astype(np.float32)
print(f"mbank FULL native bank {len(bank):,}   N_inj={len(sig):,}   k={K}"); sys.stdout.flush()

tree = cKDTree(fz(bank))
dist, idx = tree.query(fz(sig), k=K, workers=-1)
reach = dist[:, K-1] / r_target
print(f"shortlist reach / r_target: min {reach.min():.2f}  10th pct {np.percentile(reach,10):.2f}  "
      f"median {np.median(reach):.2f}   injections below {ENVELOPE} r_target: {(reach < ENVELOPE).sum()}")
sys.stdout.flush()

n = len(sig)
t1 = np.repeat(sig, K, axis=0); t2 = bank[idx.reshape(-1)]
m = np.empty(len(t1)); t0 = time.time()
for s in range(0, len(t1), CHUNK):
    e = min(s+CHUNK, len(t1))
    with torch.no_grad():
        m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e], dtype=torch.float32),
                                         torch.tensor(t2[s:e], dtype=torch.float32)))
    if (s//CHUNK) % 50 == 0:
        print(f"  {100*e/len(t1):5.1f}% [{(time.time()-t0)/60:.0f} min]"); sys.stdout.flush()
m = np.nan_to_num(m.reshape(n, K), nan=-np.inf)
run = np.maximum.accumulate(m, axis=1)
np.save('ff_mbank_full_k500_seed4002.npy', run[:, K-1])
print(f"\n{'k':>6} | {'miss [%]':>9}")
for k in [1, 20, 100, 200, 500]:
    print(f"{k:>6} | {100*(run[:, k-1] < THR).mean():9.2f}")
ff = run[:, K-1]; c = 100*(ff >= THR).mean(); se = 100*np.sqrt((c/100)*(1-c/100)/n)
print(f"\nmbank FULL {len(bank):,}: coverage {c:.2f}% +/- {se:.2f}   min FF {ff.min():.4f}")
print(f"isobank 2,267,558 (same injections, k=500): 96.64% +/- 0.18")
print(f"=> template reduction if isobank coverage >= mbank full: {100*(1-2_267_558/len(bank)):.1f}%")
