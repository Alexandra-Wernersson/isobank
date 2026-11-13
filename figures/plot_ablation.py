"""
plot_ablation.py -- the loss ablation and the weight sensitivity (eccentric 5D) as one dot plot.

    python figures/plot_ablation.py --out figures/ablation_5d.pdf

Numbers of Tables tab:ablation_5d and tab:weights_5d: banks of 2.27M templates (+/- 3 %), 4,000 injections, missed
fraction = FF < 0.97, one dot per retraining, diamond = mean. The band is the range of the five retrainings of the
full objective (2.75-3.73 %), the dashed line its mean.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ISO, MB, INK, BAND = '#0284c7', '#6b7280', '#111827', '#dbeafe'   # isobank's blue for the full objective
FULL = [3.73, 2.75, 3.60, 3.45, 3.28]
PANELS = [   # (title, [(label, miss rates per retraining)])
    ('Terms removed', [
        (r'without $\mathcal{L}_{\rm glob}$', [3.65, 3.48]),
        (r'without $\mathcal{L}_{\rm loc}$', [5.15, 4.60]),
        (r'without $\mathcal{L}_{\rm bnd}$', [5.33, 3.73, 4.25, 4.40]),
        (r'without $\mathcal{L}_{\rm edge}$', [4.90, 5.45]),
        (r'without adaptive $\gamma_i$', [3.98, 5.63, 3.53, 3.90]),
        (r'$\mathcal{L}_{\rm loc}$ and $\mathcal{L}_{\rm edge}$ only', [3.35, 4.85]),
        (r'$\mathcal{L}_{\rm glob}$ only', [6.85, 6.83]),
    ]),
    ('Weights changed', [
        (r'$\mathcal{L}_{\rm loc}\times0.5$', [3.03, 2.93]),
        (r'$\mathcal{L}_{\rm loc}\times2$', [4.33, 3.70]),
        (r'$\mathcal{L}_{\rm bnd}\times0.5$', [3.18, 2.90]),
        (r'$\mathcal{L}_{\rm bnd}\times2$', [4.23, 3.60]),
        (r'$\mathcal{L}_{\rm edge}\times0.5$', [4.13, 4.48]),
        (r'$\mathcal{L}_{\rm edge}\times2$', [3.70, 3.28]),
        ('tail weight 1', [3.28, 3.53]),
        ('tail weight 10', [5.43, 5.15]),
    ]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='figures/ablation_5d.pdf')
    args = ap.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')
    sizes = [len(rows) + 1 for _, rows in PANELS]
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.4, 5.8), sharex=True,
                             gridspec_kw=dict(height_ratios=sizes, hspace=0.12))
    for ax, (title, rows) in zip(axes, PANELS):
        rows = [('full objective', FULL)] + rows
        ys = -np.arange(len(rows))
        ax.axvspan(min(FULL), max(FULL), color=BAND, lw=0, zorder=0)
        ax.axvline(np.mean(FULL), color=ISO, ls='--', lw=1, zorder=1)
        for y, (label, v) in zip(ys, rows):
            col = ISO if label == 'full objective' else MB
            ax.plot(v, [y] * len(v), 'o', ms=6, mfc='white', mec=col, mew=1.5, zorder=3)
            ax.plot(np.mean(v), y, 'D', ms=7, color=col, mec='white', mew=0.8, zorder=4)
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
        ax.get_yticklabels()[0].set_fontweight('bold')
        ax.set_ylim(ys[-1] - 0.6, 0.6)
        ax.grid(False, axis='y')
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    axes[-1].set_xlabel('Missed injections, FF < 0.97 (%)')
    axes[-1].set_xlim(2.5, 7.2)
    axes[-1].legend(handles=[
        plt.Line2D([], [], ls='', marker='o', mfc='white', mec=INK, mew=1.5, ms=6, label='one retraining'),
        plt.Line2D([], [], ls='', marker='D', color=INK, ms=7, label='mean'),
        plt.Rectangle((0, 0), 1, 1, color=BAND, label='range of the full objective')],
        loc='center right', fontsize=7.5, frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
