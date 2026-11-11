"""
ff_cell_k500.py <cell> — one FF-table cell at k=500, on the SAME injections as
the k=200 run (seeds 4000/4001/4002/4003), plus two things that run for free:

  * the full k-convergence curve.  cKDTree returns neighbours distance-sorted,
    so FF at any k <= 500 is the running max over the first k.  Convergence is
    then verified per cell instead of assumed from one other injection set.
  * the RANK, in latent-distance order, of the template that actually gives the
    best true match.  A perfect isometry with a perfect distance->match
    relation would give rank 1 every time; the observed distribution says how
    much of the k dependence is isometry error vs. the intrinsic scatter of the
    proper-distance -> true-match relation.
"""
import os, sys, time
import numpy as np, torch
from scipy.spatial import cKDTree

sys.path.insert(0, '/gpfs/home5/awernersson/projects/isobank_testing')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import config as C
from isobank.flow import load_checkpoint

CELL  = sys.argv[1]
K, THR, CHUNK, POOL = 500, 0.97, 20_000, 2_000_000
#	optional 2nd arg: another bank; outputs then get a tag from its filename
BANK  = sys.argv[2] if len(sys.argv) > 2 else 'templates_matched_2p27M.npy'
TAG   = '' if len(sys.argv) <= 2 else '_' + os.path.splitext(os.path.basename(BANK))[0]
ENVELOPE = 1.85                 # no pair reaches match >= 0.97 beyond 1.85 r_target

flow, ckpt = load_checkpoint(C.BEST_CKPT); metric = C.make_metric()
r_target = float(ckpt['r_target'])
def fz(a, ld=False):
    zo, lo_ = [], []
    for i in range(0, len(a), 200_000):
        with torch.no_grad(): z, l = flow(torch.tensor(a[i:i+200_000], dtype=torch.float32))
        zo.append(z.numpy())
        if ld: lo_.append(l.numpy())
    return (np.vstack(zo), np.concatenate(lo_)) if ld else np.vstack(zo)

def draw_ecc(n, seed, region):
    lo, hi = C.lo.copy(), C.hi.copy()
    if region == 'BNS':  hi[0] = 3.0
    if region == 'NSBH': lo[0] = 3.0
    rng = np.random.default_rng(seed); o, h = [], 0
    while h < n:
        b = rng.uniform(lo, hi, (4*n, C.D)).astype(np.float32)
        b = b[b[:, 0] >= b[:, 1]]; o.append(b); h += len(b)
    return np.vstack(o)[:n]

def draw_ecc_mw(n, seed):
    rng = np.random.default_rng(seed); pool, have = [], 0
    while have < POOL:
        b = rng.uniform(C.lo, C.hi, (500_000, C.D)).astype(np.float32)
        b[:, 0] = np.exp(rng.uniform(np.log(C.lo[0]), np.log(C.hi[0]), len(b)))
        b[:, 1] = np.exp(rng.uniform(np.log(C.lo[1]), np.log(C.hi[1]), len(b)))
        b = b[b[:, 0] >= b[:, 1]]; pool.append(b); have += len(b)
    pool = np.vstack(pool)[:POOL].astype(np.float32)
    _, ld = fz(pool, ld=True)
    logw = ld + np.log(pool[:, 0]) + np.log(pool[:, 1])
    w = np.exp(logw - logw.max()); w /= w.sum()
    print(f"  pool ESS = {1.0/np.sum(w**2):,.0f}"); sys.stdout.flush()
    return pool[rng.choice(len(pool), size=n, replace=False, p=w)]

SPEC = {'BNS':     lambda: draw_ecc(5_000, 4000, 'BNS'),
        'NSBH':    lambda: draw_ecc(5_000, 4001, 'NSBH'),
        'ALL':     lambda: draw_ecc(10_000, 4002, 'Total'),
        'ALL-mw':  lambda: draw_ecc_mw(10_000, 4003)}
assert CELL in SPEC, f"cell must be one of {list(SPEC)}"

sig = SPEC[CELL]()
bank = np.load(BANK).astype(np.float32)
print(f"cell={CELL}  N_inj={len(sig):,}  bank={len(bank):,}  k={K}"); sys.stdout.flush()

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
rank = np.argmax(m, axis=1) + 1

np.save(f'cell_{CELL}_k{K}{TAG}_ff.npy', run[:, K-1])
np.save(f'cell_{CELL}_k{K}{TAG}_rank.npy', rank)

print(f"\n{'='*56}\ncell {CELL}: k-convergence\n{'='*56}")
print(f"{'k':>6} | {'miss FF<0.97 [%]':>17} | {'d vs prev':>10}")
prev = None
for k in [1, 5, 20, 50, 100, 200, 300, 400, 500]:
    ff = run[:, k-1]; miss = 100*(ff < THR).mean()
    print(f"{k:>6} | {miss:17.2f} | {'' if prev is None else f'{miss-prev:+.2f}':>10}"); prev = miss
se = 100*np.sqrt((prev/100)*(1-prev/100)/n)
print(f"\nFINAL cell {CELL}: miss = {prev:.2f} +/- {se:.2f} %   (k={K}, N={n:,})")
print(f"\nbest-match rank in latent-distance order:")
for q in [50, 75, 90, 95, 99]:
    print(f"   {q}th pct rank = {np.percentile(rank, q):.0f}")
print(f"   rank==1: {100*(rank==1).mean():.1f}%   rank<=20: {100*(rank<=20).mean():.1f}%"
      f"   rank<=200: {100*(rank<=200).mean():.1f}%   max={rank.max()}")
