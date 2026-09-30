"""
plot_ff_cdf.py -- cumulative distribution of fitting factors for one or more banks.

    python scripts/plot_ff_cdf.py --out figures/ff_cdf.pdf \
        --ff runs/eccentric_5d/ff.npy:"isobank (2.27M)" \
        --ff runs/eccentric_5d/ff_other.npy:"mbank (2.95M)"

Each input is a .npy of fitting factors, as written by
``evaluate_bank.py --save-ff``. The curve gives the percentage of injections
with FF below a given value, so its height at the threshold is the fraction of
missed injections; the legend reports coverage (FF >= threshold).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PALETTE = ['#0284c7', '#d97706', '#059669', '#7c3aed', '#dc2626']


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ff', action='append', required=True,
                    help='path[:label] (repeatable)')
    ap.add_argument('--out', default='ff_cdf.pdf')
    ap.add_argument('--threshold', type=float, default=0.97)
    ap.add_argument('--xlim', default='0.85,1.0')
    ap.add_argument('--dashed', default='',
                    help='comma-separated indices (0-based) to draw dashed')
    args = ap.parse_args()

    dashed = {int(i) for i in args.dashed.split(',') if i.strip()}
    lo, hi = (float(v) for v in args.xlim.split(','))
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, spec in enumerate(args.ff):
        path, _, label = spec.partition(':')
        ff = np.load(path); label = label or path
        s = np.sort(ff); y = 100 * np.arange(1, len(s) + 1) / len(s)
        cov = 100 * (ff >= args.threshold).mean()
        ax.semilogy(s, y, color=PALETTE[i % len(PALETTE)], lw=2.0,
                    ls=(0, (5, 2)) if i in dashed else '-',
                    label=f'{label} — {cov:.2f}%')
        print(f"{label}: N={len(ff):,}  coverage {cov:.2f}%  worst FF {ff.min():.4f}")
    ax.axvline(args.threshold, color='#dc2626', ls=':', lw=1.6)
    ax.set_xlabel('Fitting factor', fontsize=11)
    ax.set_ylabel('Cumulative fraction of injections (%)', fontsize=11)
    ax.set_xlim(lo, hi); ax.set_ylim(0.008, 150)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8.5, loc='upper left', framealpha=0.92)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()
