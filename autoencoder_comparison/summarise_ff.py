"""
summarise_ff.py -- the spread of the fitting factors of one or more banks.

    python autoencoder_comparison/summarise_ff.py \
        runs/eccentric_5d/ff_isobank.npy:isobank \
        autoencoder_comparison/runs/eccentric_5d/ff_latent5_encoder.npy:"autoencoder, encoder loss"

Each input is a .npy of fitting factors (evaluate_bank.py --save-ff). For each
bank: coverage (FF >= threshold) with its binomial error, mean and standard
deviation, median, the 1st, 5th and 10th percentiles, and the worst value.
The worst value is the minimum of one injection set and changes a lot from
set to set; the low percentiles describe the poorly covered tail more stably.
The table is printed as markdown.
"""
import argparse
import numpy as np


def summary(ff, threshold):
    p = (ff >= threshold).mean()
    p1, p5, p10, p50 = np.percentile(ff, [1, 5, 10, 50])
    return dict(n=len(ff), coverage=100 * p, coverage_err=100 * np.sqrt(p * (1 - p) / len(ff)),
                mean=ff.mean(), std=ff.std(ddof=1), p1=p1, p5=p5, p10=p10, median=p50, worst=ff.min())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ff', nargs='+', help='path[:label]')
    ap.add_argument('--threshold', type=float, default=0.97)
    args = ap.parse_args()

    print(f"| Bank | N | Coverage, FF ≥ {args.threshold} | FF mean ± std | median | 1st / 5th / 10th pct | worst |")
    print("|---|---|---|---|---|---|---|")
    for spec in args.ff:
        path, _, label = spec.partition(':')
        s = summary(np.load(path), args.threshold)
        print(f"| {label or path} | {s['n']:,} | {s['coverage']:.2f} ± {s['coverage_err']:.2f} % "
              f"| {s['mean']:.4f} ± {s['std']:.4f} | {s['median']:.4f} "
              f"| {s['p1']:.4f} / {s['p5']:.4f} / {s['p10']:.4f} | {s['worst']:.4f} |")


if __name__ == '__main__':
    main()
