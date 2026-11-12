"""
ff_parallel.py -- scripts/evaluate_bank.py split over independent jobs.

    python autoencoder_comparison/ff_parallel.py part --config configs/eccentric_5d.py \
        --bank BANK.npy --parts 50 --part 7 --out-dir PARTS_DIR
    python autoencoder_comparison/ff_parallel.py join --out-dir PARTS_DIR --save-ff FF.npy

Every part draws the same injections as evaluate_bank.py (same seed, same
validity mask) and scores its own slice of them; joining the parts gives the
same fitting factors as one evaluate_bank.py run, and prints the same summary
(shortlist reach, miss rate against k, coverage).
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import load_checkpoint

KS = [1, 5, 20, 50, 100, 200, 500, 1000, 2000]      # the k of evaluate_bank.py's table


def load_config(path):
    spec = importlib.util.spec_from_file_location('case_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def injections(C, n_inj, seed):
    """As evaluate_bank.py: uniform in the coordinates, validity mask applied."""
    rng = np.random.default_rng(seed)
    draws, have = [], 0
    while have < n_inj:
        b = rng.uniform(C.lo, C.hi, (4 * n_inj, C.D)).astype(np.float32)
        if C.valid_mask is not None:
            b = b[C.valid_mask(b)]
        draws.append(b); have += len(b)
    return np.vstack(draws)[:n_inj]


def part(args):
    C = load_config(args.config)
    flow, ckpt = load_checkpoint(C.BEST_CKPT)
    metric = C.make_metric()
    r_target = float(ckpt['r_target'])

    def to_latent(a):
        out = []
        for i in range(0, len(a), 200_000):
            with torch.no_grad():
                z, _ = flow(torch.tensor(a[i:i+200_000], dtype=torch.float32))
            out.append(z.numpy())
        return np.vstack(out)

    sig = injections(C, args.n_inj, args.seed)
    lo, hi = np.linspace(0, len(sig), args.parts + 1).astype(int)[[args.part, args.part + 1]]
    sig = sig[lo:hi]
    bank = np.load(args.bank).astype(np.float32)
    k = min(args.k, len(bank))
    print(f"{C.CASE_NAME}\nbank {len(bank):,} templates   injections {lo}:{hi} of {args.n_inj:,}   k = {k}")
    sys.stdout.flush()

    t0 = time.time()
    dist, idx = cKDTree(to_latent(bank)).query(to_latent(sig), k=k, workers=-1)
    reach = (dist[:, -1] if k > 1 else dist) / r_target
    print(f"shortlist: {(time.time() - t0) / 60:.1f} min"); sys.stdout.flush()

    t1 = np.repeat(sig, k, axis=0)
    t2 = bank[np.asarray(idx).reshape(-1)]
    m = np.empty(len(t1)); t0 = time.time()
    for s in range(0, len(t1), args.chunk):
        e = min(s + args.chunk, len(t1))
        with torch.no_grad():
            m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e], dtype=torch.float32),
                                             torch.tensor(t2[s:e], dtype=torch.float32)))
        print(f"  {100*e/len(t1):5.1f}%  [{(time.time()-t0)/60:.0f} min]"); sys.stdout.flush()

    m = np.nan_to_num(m.reshape(len(sig), k), nan=-np.inf)
    running = np.maximum.accumulate(m, axis=1)
    ks = np.array([kk for kk in KS if kk <= k])
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, f'part_{args.part:04d}.npz'), ff=running[:, -1],
             ff_at_k=running[:, ks - 1], ks=ks, reach=reach, n_bank=len(bank), k=k)


def join(args):
    if args.parts is None:      # all the parts in out_dir, which must be 0 ... n-1
        found = sorted(f for f in os.listdir(args.out_dir) if f.startswith('part_') and f.endswith('.npz'))
        args.parts = len(found)
        if [int(f[5:9]) for f in found] != list(range(args.parts)):
            raise SystemExit(f"parts in {args.out_dir} are not 0 ... {args.parts - 1}")
    files = [os.path.join(args.out_dir, f'part_{i:04d}.npz') for i in range(args.parts)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f"{len(missing)} parts missing, e.g. {missing[0]}")
    parts = [np.load(f, allow_pickle=False) for f in files]
    ff = np.concatenate([p['ff'] for p in parts])
    ff_at_k = np.concatenate([p['ff_at_k'] for p in parts])
    reach = np.concatenate([p['reach'] for p in parts])
    ks, n_bank, k = parts[0]['ks'], int(parts[0]['n_bank']), int(parts[0]['k'])

    print(f"bank {n_bank:,} templates   injections {len(ff):,}   k = {k}   ({args.parts} parts)")
    print(f"shortlist reach / r_target: min {reach.min():.2f}  "
          f"10th pct {np.percentile(reach, 10):.2f}  median {np.median(reach):.2f}")
    if args.save_ff:
        np.save(args.save_ff, ff)
    print(f"\n{'k':>6} | {'missed [%]':>11}")
    for j, kk in enumerate(ks):
        print(f"{kk:>6} | {100*(ff_at_k[:, j] < args.threshold).mean():11.2f}")
    p = (ff >= args.threshold).mean(); se = np.sqrt(p * (1 - p) / len(ff))
    print(f"\ncoverage (FF >= {args.threshold}): {100*p:.2f} +/- {100*se:.2f} %"
          f"   worst FF {ff.min():.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    p = sub.add_parser('part')
    p.add_argument('--config', required=True)
    p.add_argument('--bank', required=True)
    p.add_argument('--part', type=int, required=True)
    p.add_argument('--n-inj', type=int, default=10_000)
    p.add_argument('--k', type=int, default=500)
    p.add_argument('--seed', type=int, default=4002)
    p.add_argument('--chunk', type=int, default=20_000)
    j = sub.add_parser('join')
    j.add_argument('--threshold', type=float, default=0.97)
    j.add_argument('--save-ff', default=None)
    p.add_argument('--parts', type=int, required=True)
    j.add_argument('--parts', type=int, default=None, help='default: all the parts in --out-dir')
    for q in (p, j):
        q.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    part(args) if args.mode == 'part' else join(args)


if __name__ == '__main__':
    main()
