"""
plot_tuning_violins.py -- the mbank tuning sweep, one column per variant, isobank first: coverage and bank size
on top, the distribution of the fitting factor underneath.

    python mbank_comparison/tuning/plot_tuning_violins.py --ff-dir mbank_comparison/tuning/results/ff \
        --isobank-ff runs/eccentric_5d/ff_isobank.npy --isobank-size 3009368

--ff-dir holds <variant>.npy, the fitting factors of each variant (WORK/<variant>/ff.npy at Nikhef), on the
10,000 injections of seed 4002, as isobank's. Top: coverage (FF >= 0.97, +/- 1 sigma, filled, left axis) and
templates (open, right axis). Bottom: violins of FF, cut at FLOOR
(the share of injections below it is printed); the dashed line is FF = 0.97. Colour = what was changed.
Writes results/tuning_violins.pdf (or --out).
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ISO, PAPER, INK = '#0284c7', '#8b8f98', '#111827'   # isobank's blue, as in every other figure
FLOOR = 0.96        # bottom of the FF axis
GROUPS = [   # (legend entry, colour, [(variant, tick label)])
    ('mbank, paper settings', PAPER, [('paper_a', 'run 1'), ('paper_b', 'run 2'), ('paper_c', 'run 3')]),
    ('loss weighting (paper: none)', '#cc6677', [('weighted_0.5', 'exp.\n0.5'), ('weighted_1', 'exp.\n1')]),
    ('start of C (paper: 0.9 quantile)', '#ddcc77', [('quantile_0.5', '0.5'), ('quantile_0.99', '0.99')]),
    ('box margin (paper: 0.01)', '#117733', [('padding_0.001', '0.001'), ('padding_0.05', '0.05')]),
    ('optimiser (paper: lr 5e-3, batch 5k)', '#aa4499', [('lr_0.0005', 'lr\n5e-4'), ('batch_1000', 'batch\n1k')]),
    ('training points (paper: 15k)', '#999933', [('data_50k', '50k\npoints')]),
    ('importance-sampling placement', '#882255', [('is_wmax_10', 'W\n= 10'), ('is_no_clip', 'no\nclip')]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ff-dir', required=True)
    ap.add_argument('--isobank-ff', required=True)
    ap.add_argument('--csv', default=None, help='tuning.csv, for the bank sizes (default: next to --ff-dir)')
    ap.add_argument('--isobank-bank', default=None, help="isobank's bank, for its size")
    ap.add_argument('--isobank-size', type=int, default=None, help="isobank's size, without --isobank-bank")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(os.path.dirname(os.path.abspath(args.ff_dir)), 'tuning.csv')
    sizes = {r['name']: int(r['templates']) for r in csv.DictReader(open(csv_path))} if os.path.exists(csv_path) else {}
    if args.isobank_bank:
        sizes['isobank'] = len(np.load(args.isobank_bank, mmap_mode='r'))
    elif args.isobank_size:
        sizes['isobank'] = args.isobank_size
    items = [('isobank', 'isobank', ISO, np.load(args.isobank_ff))]   # (name, tick label, colour, ff)
    xs, x = [0.], 0.
    for _, col, members in GROUPS:
        x += 0.5
        for name, label in members:
            x += 1.; xs.append(x)
            items.append((name, label, col, np.load(os.path.join(args.ff_dir, f'{name}.npy'))))
    xs = np.array(xs)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (a_cov, a_ff) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                                      gridspec_kw=dict(height_ratios=[1.4, 3], hspace=0.08))
    a_n = a_cov.twinx()   # coverage (filled, left) below, templates (open, right) above; the ticks share gridlines
    cov = np.array([100 * (ff >= .97).mean() for *_, ff in items])
    err = np.array([100 * np.sqrt(c / 100 * (1 - c / 100) / len(ff)) for c, (*_, ff) in zip(cov, items)])
    n = np.array([sizes.get(name, np.nan) / 1e6 for name, *_ in items])
    cols = [col for _, _, col, _ in items]
    for xi, c, e, ni, col in zip(xs, cov, err, n, cols):
        a_cov.errorbar(xi, c, yerr=e, fmt='o', color=col, ms=6.5, lw=1.4, mec='white', mew=1.0, zorder=3)
        a_n.plot(xi, ni, 's', mfc='white', mec=col, ms=5.5, mew=1.6, zorder=3)
    a_cov.annotate(f'{cov[0]:.1f}%', (xs[0], cov[0]), xytext=(9, 0), textcoords='offset points', va='center',
                   fontsize=8.5, color=INK)
    a_cov.set_ylim(92, 100); a_cov.set_yticks([92, 96, 100]); a_cov.set_ylabel('Coverage (%)')
    a_n.set_ylim(2.0, 3.6); a_n.set_yticks([2.0, 2.8, 3.6]); a_n.set_ylabel('Templates (M)')
    a_n.grid(False); a_n.tick_params(axis='y', labelsize=8.5)
    for sp in a_n.spines.values():
        sp.set_visible(False)
    a_cov.legend(handles=[plt.Line2D([], [], ls='', marker='o', color=INK, mec='white', ms=6.5, label='coverage'),
                          plt.Line2D([], [], ls='', marker='s', mfc='white', mec=INK, mew=1.6, ms=5.5,
                                     label='templates')],
                 loc='upper right', ncol=2, borderaxespad=0.3, fontsize=8, frameon=False, handletextpad=0.3)

    below = []
    for (name, _, col, ff), xi in zip(items, xs):
        v = a_ff.violinplot(ff, positions=[xi], widths=0.82, showextrema=False, points=500, bw_method=0.15)
        for body in v['bodies']:
            body.set_facecolor(col); body.set_edgecolor('white'); body.set_linewidth(0.8); body.set_alpha(0.9)
        a_ff.plot([xi - 0.2, xi + 0.2], [np.median(ff)] * 2, color='white', lw=1.6, solid_capstyle='round')
        below.append((name, 100 * (ff < FLOOR).mean()))
    a_ff.axhline(0.97, color=INK, ls='--', lw=1.1, zorder=4)
    a_ff.text(1.005, 0.97, 'FF = 0.97', transform=a_ff.get_yaxis_transform(), color=INK, fontsize=8, va='center')
    a_ff.set_ylim(FLOOR, 1.0005); a_ff.set_yticks([0.96, 0.97, 0.98, 0.99, 1.0])
    a_ff.set_ylabel('Fitting factor')
    a_ff.set_xticks(xs, [it[1] for it in items], fontsize=8.5)
    a_ff.get_xticklabels()[0].set_fontweight('bold')
    a_ff.set_xlim(xs[0] - 0.6, xs[-1] + 0.6)
    for ax in (a_cov, a_ff):
        ax.grid(False, axis='x'); ax.tick_params(axis='y', labelsize=8.5)
    fig.align_ylabels((a_cov, a_ff))
    handles = [Patch(color=ISO, label='isobank')] + [Patch(color=col, label=lab) for lab, col, _ in GROUPS]
    a_ff.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=4, fontsize=8, frameon=False,
                handlelength=1.2, columnspacing=1.4)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ff_dir)), 'tuning_violins.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {out}')
    print('below FF = %.2f: ' % FLOOR + ', '.join(f'{nm} {b:.2f}%' for nm, b in below))


if __name__ == '__main__':
    main()
