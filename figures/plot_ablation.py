"""
plot_ablation.py -- the loss ablation and the coefficient sensitivity (eccentric 5D) in one dot plot.

    python figures/plot_ablation.py --out figures/ablation_5d.pdf

Numbers of Tables tab:ablation_5d and tab:weights_5d: banks of 2.27M templates (+/- 3 %), 4,000 injections, missed
fraction = FF < 0.97. Dot = mean of the retrainings of each setting, bar = their standard deviation (n - 1; from
two retrainings only a rough estimate). The band is the mean +/- standard deviation of the full objective, the dashed
line its mean. Colour: terms removed or coefficients changed; gamma is the tail weight (nominal 3; 0 = no tail weighting).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ISO, INK, BAND = '#0284c7', '#111827', '#dbeafe'   # isobank's blue for the full objective
FULL = [3.73, 2.75, 3.60, 3.45, 3.28]
GROUPS = [   # (legend entry, colour, [(label, miss rates per retraining)])
    ('terms removed', 'darkorange', [
        (r'without $\mathcal{L}_{\rm glob}$', [3.65, 3.48]),
        (r'without $\mathcal{L}_{\rm loc}$', [5.15, 4.60]),
        (r'without $\mathcal{L}_{\rm bnd}$', [5.33, 3.73, 4.25, 4.40]),
        (r'without $\mathcal{L}_{\rm edge}$', [4.90, 5.45]),
        (r'$\mathcal{L}_{\rm loc}$ and $\mathcal{L}_{\rm edge}$ only', [3.35, 4.85]),
        (r'$\mathcal{L}_{\rm glob}$ only', [6.85, 6.83]),
    ]),
    ('coefficients changed', 'mediumseagreen', [
        (r'$\beta_{loc}\times0.5$', [3.03, 2.93]),
        (r'$\beta_{loc}\times2$', [4.33, 3.70]),
        (r'$\beta_{bnd}\times0.5$', [3.18, 2.90]),
        (r'$\beta_{bnd}\times2$', [4.23, 3.60]),
        (r'$\beta_{edge}\times0.5$', [4.13, 4.48]),
        (r'$\beta_{edge}\times2$', [3.70, 3.28]),
        (r'$\gamma=0$', [3.98, 5.63, 3.53, 3.90]),
        (r'$\gamma=1$', [3.28, 3.53]),
        (r'$\gamma=10$', [5.43, 5.15]),
    ]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='figures/ablation_5d.pdf')
    args = ap.parse_args()

    rows, y = [('full objective', ISO, FULL, 0.)], 0.   # (label, colour, values, y), a gap before each group
    for _, col, members in GROUPS:
        y -= 0.5
        for label, v in members:
            y -= 1.; rows.append((label, col, v, y))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    m_full, s_full = np.mean(FULL), np.std(FULL, ddof=1)
    ax.axvspan(m_full - s_full, m_full + s_full, color=BAND, lw=0, zorder=0)
    ax.axvline(m_full, color=ISO, ls='--', lw=1, zorder=1)
    for label, col, v, yi in rows:
        ax.errorbar(np.mean(v), yi, xerr=np.std(v, ddof=1), fmt='o', ms=5.5, color=col, mec='white', mew=0.7,
                    lw=1.4, capsize=2, zorder=4)
    ax.set_yticks([r[3] for r in rows], [r[0] for r in rows], fontsize=8)
    ax.set_ylim(rows[-1][3] - 0.6, 0.6); ax.tick_params(axis='x', labelsize=8.5)
    ax.grid(False, axis='y')
    ax.set_xlabel('Missed injections, FF < 0.97 (%)', fontsize=9)
    ax.set_xlim(2.5, 7.2)
    handles = [plt.Line2D([], [], ls='-', marker='o', color=INK, ms=5.5, lw=1.4, label=r'$\mu \pm \sigma$')]
    handles += [plt.Line2D([], [], ls='', marker='o', color=col, ms=5.5, label=lab)
                for lab, col in [('full objective', ISO)] + [(g[0], g[1]) for g in GROUPS]]
    ax.legend(handles=handles, loc='upper right', ncol=2, fontsize=7, columnspacing=1.0, handletextpad=0.4, frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
