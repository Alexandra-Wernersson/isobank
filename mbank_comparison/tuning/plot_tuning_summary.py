"""
plot_tuning_summary.py -- the mbank tuning sweep in one reviewer-facing figure.

    python mbank_comparison/tuning/plot_tuning_summary.py --csv mbank_comparison/tuning/results/tuning.csv \
        --isobank-ff runs/eccentric_5d/ff_isobank.npy --isobank-bank runs/eccentric_5d/bank_isobank.npy

One row per variant of variants.tsv, grouped by what was changed: coverage (FF >= 0.97, +/- 1 sigma) on the
left, bank size on the right, isobank on top. The shaded band is the range of the three runs with the paper's
settings, i.e. the run-to-run scatter. Writes results/tuning_summary.pdf next to the csv.
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ISO, MB, BAND, INK, MUTED, GRID = '#0284c7', '#475569', '#e2e8f0', '#111827', '#6b7280', '#e5e7eb'
GROUPS = [   # (heading, [(variant, label)])
    ('paper settings', [('paper_a', 'run 1'), ('paper_b', 'run 2'), ('paper_c', 'run 3')]),
    ('loss', [('weighted_0.5', 'weighted, exponent 0.5 (mbank\'s)'), ('weighted_1', 'weighted, exponent 1')]),
    ('start of the constant C', [('quantile_0.5', '0.5 quantile (paper 0.9)'), ('quantile_0.99', '0.99 quantile')]),
    ('margin around the box', [('padding_0.001', '0.001 (paper 0.01)'), ('padding_0.05', '0.05')]),
    ('optimiser', [('lr_0.0005', 'learning rate 5e-4 (paper 5e-3)'), ('batch_1000', 'batch 1,000 (paper 5,000)')]),
    ('training data', [('data_50k', '50,000 points (paper 15,000)')]),
    ('placement', [('is_wmax_10', 'importance sampling, W_max = 10'), ('is_no_clip', 'importance sampling, no clip')]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--isobank-ff', required=True, help='evaluate_bank.py fitting factors, the same 10,000 injections')
    ap.add_argument('--isobank-bank', required=True)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    rows = {r['name']: r for r in csv.DictReader(open(args.csv))}
    ff = np.load(args.isobank_ff); p = (ff >= .97).mean()
    iso = dict(n=len(np.load(args.isobank_bank, mmap_mode='r')) / 1e6, cov=100 * p,
               err=100 * np.sqrt(p * (1 - p) / len(ff)))

    # y positions: isobank on top, then the groups with a gap and a heading each
    labels, ys, heads, y = [], [], [], 0.
    items = []
    for head, members in GROUPS:
        y -= 0.9; heads.append((y, head)); y -= 0.75
        for name, label in members:
            items.append((y, rows[name])); labels.append(label); ys.append(y); y -= 1.
    y_end = y
    paper = [float(rows[n]['coverage']) for n in ('paper_a', 'paper_b', 'paper_c')]
    paper_n = [float(rows[n]['templates']) / 1e6 for n in ('paper_a', 'paper_b', 'paper_c')]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.2, 6.6), sharey=True, gridspec_kw=dict(width_ratios=[1.6, 1]))
    for ax, lo, hi in ((a1, min(paper), max(paper)), (a2, min(paper_n), max(paper_n))):
        ax.axvspan(lo, hi, color=BAND, zorder=0, lw=0)
        ax.grid(True, axis='x', color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ('top', 'right', 'left'):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis='y', length=0)
    for yi, r in items:
        a1.errorbar(float(r['coverage']), yi, xerr=float(r['error']), fmt='o', color=MB, ms=6, lw=1.4, capsize=0,
                    mec='white', mew=1.2, zorder=3)
        a2.plot(float(r['templates']) / 1e6, yi, 'o', color=MB, ms=6, mec='white', mew=1.2, zorder=3)
    y_iso = 0.6
    a1.errorbar(iso['cov'], y_iso, xerr=iso['err'], fmt='D', color=ISO, ms=7, lw=1.4, capsize=0, mec='white',
                mew=1.2, zorder=4)
    a2.plot(iso['n'], y_iso, 'D', color=ISO, ms=7, mec='white', mew=1.2, zorder=4)
    a1.annotate(f"{iso['cov']:.2f} %", (iso['cov'], y_iso), xytext=(-8, 0), textcoords='offset points',
                ha='right', va='center', fontsize=8.5, color=INK)

    a1.set_yticks([y_iso] + ys, ['isobank'] + labels, fontsize=8.5)
    a1.get_yticklabels()[0].set_fontweight('bold')
    for yh, head in heads:
        a1.text(-0.01, yh + 0.15, head, transform=a1.get_yaxis_transform(), ha='right', va='center',
                fontsize=8.5, color=MUTED, style='italic')
    a1.set_xlim(92.5, 99.2); a2.set_xlim(2.75, 3.2)
    a1.set_xlabel('coverage, FF >= 0.97 (%)'); a2.set_xlabel('templates (millions)')
    a1.set_ylim(y_end + 0.3, y_iso + 0.9)
    a1.text(max(paper) + 0.12, heads[0][0] + 0.15, 'shaded: range of the 3 runs\nwith the paper\'s settings',
            fontsize=7.5, color=MUTED, va='center', ha='left')
    fig.suptitle('mbank retrained with one change at a time (eccentric 5D, 10,000 injections)', fontsize=10.5,
                 color=INK, x=0.02, ha='left')
    plt.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.csv)), 'tuning_summary.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
