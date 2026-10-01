"""
evaluate_bank.py -- measure the coverage of a bank with true waveform matches.

    python scripts/evaluate_bank.py --config configs/eccentric_5d.py \
        --bank runs/eccentric_5d/bank.npy --n-inj 10000 --k 500

Injections are drawn uniformly in the coordinates of the parameter space
(respecting the config's validity mask), independently of the bank. For each
injection the fitting factor is the maximum true waveform match over the bank.

A brute force search over every template is too expensive for large banks, so
exact matches are computed only against the k templates nearest to the
injection in the flow's latent space. The latent space is used solely to
shortlist candidates: every reported match is exact. The script prints

  * the shortlist reach (distance to the k-th candidate, in units of r_target),
    which must exceed the separation beyond which no pair can reach the
    threshold, and
  * the miss rate as a function of k, which must be flat at the chosen k;
    if it is still falling, the quoted coverage is an upper bound on the miss
    rate and k should be increased.
"""
import argparse, importlib.util, os, sys, time
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import load_checkpoint


def load_config(path):
    spec = importlib.util.spec_from_file_location('case_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--bank', required=True, help='bank .npy from generate_bank.py')
    ap.add_argument('--n-inj', type=int, default=10_000)
    ap.add_argument('--k', type=int, default=500, help='shortlist size (default 500)')
    ap.add_argument('--seed', type=int, default=4002)
    ap.add_argument('--threshold', type=float, default=0.97, help='minimal match')
    ap.add_argument('--chunk', type=int, default=20_000)
    ap.add_argument('--save-ff', default=None, help='optional .npy for the fitting factors')
    args = ap.parse_args()

    C = load_config(args.config)
    if not os.path.exists(C.BEST_CKPT):
        raise SystemExit(
            f"no trained flow at {C.BEST_CKPT}\n"
            f"run:  python scripts/train_flow.py --config {args.config}")
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

    #	injections: uniform in the coordinates, validity mask applied
    rng = np.random.default_rng(args.seed)
    draws, have = [], 0
    while have < args.n_inj:
        b = rng.uniform(C.lo, C.hi, (4 * args.n_inj, C.D)).astype(np.float32)
        if C.valid_mask is not None:
            b = b[C.valid_mask(b)]
        draws.append(b); have += len(b)
    sig = np.vstack(draws)[:args.n_inj]

    bank = np.load(args.bank).astype(np.float32)
    k = min(args.k, len(bank))
    print(f"{C.CASE_NAME}\nbank {len(bank):,} templates   injections {len(sig):,}   k = {k}")
    sys.stdout.flush()

    dist, idx = cKDTree(to_latent(bank)).query(to_latent(sig), k=k, workers=-1)
    reach = (dist[:, -1] if k > 1 else dist) / r_target
    print(f"shortlist reach / r_target: min {reach.min():.2f}  "
          f"10th pct {np.percentile(reach, 10):.2f}  median {np.median(reach):.2f}")
    sys.stdout.flush()

    t1 = np.repeat(sig, k, axis=0)
    t2 = bank[np.asarray(idx).reshape(-1)]
    m = np.empty(len(t1)); t0 = time.time()
    for s in range(0, len(t1), args.chunk):
        e = min(s + args.chunk, len(t1))
        with torch.no_grad():
            m[s:e] = np.asarray(metric.match(torch.tensor(t1[s:e], dtype=torch.float32),
                                             torch.tensor(t2[s:e], dtype=torch.float32)))
        if (s // args.chunk) % 50 == 0:
            print(f"  {100*e/len(t1):5.1f}%  [{(time.time()-t0)/60:.0f} min]"); sys.stdout.flush()

    m = np.nan_to_num(m.reshape(len(sig), k), nan=-np.inf)
    running = np.maximum.accumulate(m, axis=1)       # FF for every shortlist size <= k
    ff = running[:, -1]
    if args.save_ff:
        np.save(args.save_ff, ff)

    print(f"\n{'k':>6} | {'missed [%]':>11}")
    for kk in [1, 5, 20, 50, 100, 200, 500, 1000, 2000]:
        if kk <= k:
            print(f"{kk:>6} | {100*(running[:, kk-1] < args.threshold).mean():11.2f}")
    p = (ff >= args.threshold).mean(); se = np.sqrt(p * (1 - p) / len(ff))
    print(f"\ncoverage (FF >= {args.threshold}): {100*p:.2f} +/- {100*se:.2f} %"
          f"   worst FF {ff.min():.4f}")


if __name__ == '__main__':
    main()
