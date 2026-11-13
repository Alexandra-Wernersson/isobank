"""
plot_tuning_violins.py -- the mbank tuning sweep as violins of the fitting factor, one per variant, isobank first.

    python mbank_comparison/tuning/plot_tuning_violins.py --ff-dir mbank_comparison/tuning/results/ff \
        --isobank-ff runs/eccentric_5d/ff_isobank.npy

--ff-dir holds <variant>.npy, the fitting factors of each variant (WORK/<variant>/ff.npy at Nikhef), on the
10,000 injections of seed 4002, as isobank's. Each violin is the distribution of FF (higher is better; the
0.1-0.7 % of injections below FLOOR are drawn at FLOOR); the dotted line is FF = 0.97, and the numbers above each
violin are its coverage and bank size.
Writes results/tuning_violins.pdf (or --out).
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ISO, MB, INK, MUTED, GRID, RED = '#0284c7', '#64748b', '#111827', '#6b7280', '#e5e7eb', '#dc2626'
FLOOR = 0.93        # bottom of the FF axis
GROUPS = [   # (heading, [(variant, label)]), as plot_tuning_summary.py
    ('paper settings', [('paper_a', 'run 1'), ('paper_b', 'run 2'), ('paper_c', 'run 3')]),
    ('loss', [('weighted_0.5', 'weight\n^0.5'), ('weighted_1', 'weight\n^1')]),
    ('start of C (quantile)', [('quantile_0.5', '0.5'), ('quantile_0.99', '0.99')]),
    ('box margin', [('padding_0.001', '0.001'), ('padding_0.05', '0.05')]),
    ('optimiser', [('lr_0.0005', 'lr\n5e-4'), ('batch_1000', 'batch\n1,000')]),
    ('data', [('data_50k', '50k\npoints')]),
    ('placement', [('is_wmax_10', 'IS\nW=10'), ('is_no_clip', 'IS\nno clip')]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ff-dir', required=True)
    ap.add_argument('--isobank-ff', required=True)
    ap.add_argument('--csv', default=None, help='tuning.csv, for the bank sizes (default: next to --ff-dir)')
    ap.add_argument('--isobank-bank', default=None, help="isobank's bank, for its size")
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(os.path.dirname(os.path.abspath(args.ff_dir)), 'tuning.csv')
    sizes = {r['name']: int(r['templates']) for r in csv.DictReader(open(csv_path))} if os.path.exists(csv_path) else {}
    if args.isobank_bank:
        sizes['isobank'] = len(np.load(args.isobank_bank, mmap_mode='r'))
    items = [('isobank', 'isobank', None, np.load(args.isobank_ff))]
    for head, members in GROUPS:
        for name, label in members:
            items.append((name, label, head, np.load(os.path.join(args.ff_dir, f'{name}.npy'))))

    # x positions: isobank, a gap, then the groups with a small gap between them
    xs, x, prev = [], 0., None
    for name, _, head, _ in items:
        if head != prev and prev is not None or (head is not None and prev is None):
            x += 0.6
        xs.append(x); x += 1.; prev = head

    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    for (name, label, head, ff), xi in zip(items, xs):
        mm = np.clip(ff, FLOOR, None)
        col = ISO if head is None else MB
        v = ax.violinplot(mm, positions=[xi], widths=0.82, showextrema=False, points=400, bw_method=0.15)
        for body in v['bodies']:
            body.set_facecolor(col); body.set_edgecolor('white'); body.set_linewidth(0.8); body.set_alpha(0.85)
        ax.plot([xi - 0.22, xi + 0.22], [np.median(mm)] * 2, color='white', lw=1.6, solid_capstyle='round')
        cov = 100 * (ff >= .97).mean()
        ax.text(xi, 1.0035, f'{cov:.1f}', ha='center', va='bottom', fontsize=7.8,
                color=INK, fontweight='bold' if head is None else 'normal')
        if name in sizes:
            ax.text(xi, 1.0032, f'{sizes[name] / 1e6:.2f}M', ha='center', va='top', fontsize=6.8, color=MUTED)
    ax.axhline(0.97, color=RED, ls=':', lw=1.3, zorder=4)
    ax.text(xs[-1] + 0.5, 0.9705, 'FF = 0.97', color=RED, fontsize=7.8, va='bottom', ha='left')

    ax.set_xticks(xs, [it[1] for it in items], fontsize=7.6)
    ax.get_xticklabels()[0].set_fontweight('bold')
    heads, prev = [], None
    for (_, _, head, _), xi in zip(items, xs):
        if head is not None and head != prev:
            members = [x2 for (_, _, h2, _), x2 in zip(items, xs) if h2 == head]
            heads.append((np.mean(members), head, members[0] - 0.45, members[-1] + 0.45))
        prev = head
    for xm, head, a, b in heads:
        ax.annotate('', xy=(a, -0.22), xytext=(b, -0.22), xycoords=('data', 'axes fraction'),
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
        ax.text(xm, -0.27, head, transform=ax.get_xaxis_transform(), ha='center', va='top', fontsize=7.8,
                color=MUTED, style='italic')
    ax.set_yticks([0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0])
    ax.set_ylim(FLOOR - 0.002, 1.009)
    ax.set_ylabel('fitting factor (higher is better)')
    ax.text(-0.01, 1.0, 'coverage (%)\ntemplates', transform=ax.transAxes, ha='right', va='bottom', fontsize=7.8,
            color=MUTED)
    ax.set_xlim(xs[0] - 0.7, xs[-1] + 1.6)
    ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_title('mbank retrained with one change at a time, against isobank (eccentric 5D, 10,000 injections)',
                 fontsize=10, color=INK, loc='left')
    plt.tight_layout()
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.ff_dir)), 'tuning_violins.pdf')
    plt.savefig(out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {out}')


if __name__ == '__main__':
    main()
