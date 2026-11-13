"""
plot_ablation.py -- the loss ablation and the weight sensitivity (eccentric 5D) as one dot plot.

    python figures/plot_ablation.py --out figures/ablation_5d.pdf

Numbers of Tables tab:ablation_5d and tab:weights_5d: banks of 2.27M templates (+/- 3 %), 4,000 injections, missed
fraction = FF < 0.97. Diamond = mean of the retrainings, bar = their standard deviation (n - 1; from two retrainings
only a rough estimate). The band is the mean +/- standard deviation of the full objective (SIGMA), the dashed line its
mean. On the right, the difference of the means (as printed in the tables) +/- SIGMA * sqrt(1/n + 1/5), the
uncertainty of a difference of means with SIGMA as the retraining spread; it is in bold, and the row dark, when it
exceeds twice that.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ISO, CLEAR, FAINT, INK, BAND = '#0284c7', '#4b5563', '#b6bcc6', '#111827', '#dbeafe'   # isobank's blue: full objective
FULL = [3.73, 2.75, 3.60, 3.45, 3.28]
SIGMA = 0.38
PANELS = [   # (title, [(label, miss rates per retraining, difference of the means as in the tables)])
    ('Terms removed', [
        (r'without $\mathcal{L}_{\rm glob}$', [3.65, 3.48], +0.20),
        (r'without $\mathcal{L}_{\rm loc}$', [5.15, 4.60], +1.52),
        (r'without $\mathcal{L}_{\rm bnd}$', [5.33, 3.73, 4.25, 4.40], +1.07),
        (r'without $\mathcal{L}_{\rm edge}$', [4.90, 5.45], +1.82),
        (r'without adaptive $\gamma_i$', [3.98, 5.63, 3.53, 3.90], +0.90),
        (r'$\mathcal{L}_{\rm loc}$ and $\mathcal{L}_{\rm edge}$ only', [3.35, 4.85], +0.74),
        (r'$\mathcal{L}_{\rm glob}$ only', [6.85, 6.83], +3.48),
    ]),
    ('Coefficients changed', [
        (r'$\beta_{loc}\times0.5$', [3.03, 2.93], -0.38),
        (r'$\beta_{loc}\times2$', [4.33, 3.70], +0.65),
        (r'$\beta_{bnd}\times0.5$', [3.18, 2.90], -0.32),
        (r'$\beta_{bnd}\times2$', [4.23, 3.60], +0.55),
        (r'$\beta_{edge}\times0.5$', [4.13, 4.48], +0.94),
        (r'$\beta_{edge}\times2$', [3.70, 3.28], +0.13),
        ('tail weight 1', [3.28, 3.53], +0.04),
        ('tail weight 10', [5.43, 5.15], +1.93),
    ]),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='figures/ablation_5d.pdf')
    args = ap.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')
    sizes = [len(rows) + 1 for _, rows in PANELS]
    fig, axes = plt.subplots(len(PANELS), 1, figsize=(6.4, 5.8), sharex=True,
                             gridspec_kw=dict(height_ratios=sizes, hspace=0.14))
    m_full = np.mean(FULL)
    for ax, (title, rows) in zip(axes, PANELS):
        rows = [('full objective', FULL, 0.)] + rows
        ys = -np.arange(len(rows))
        ax.axvspan(m_full - SIGMA, m_full + SIGMA, color=BAND, lw=0, zorder=0)
        ax.axvline(m_full, color=ISO, ls='--', lw=1, zorder=1)
        for y, (label, v, diff) in zip(ys, rows):
            n, m = len(v), np.mean(v)
            full = label == 'full objective'
            err = SIGMA * np.sqrt(1 / n + 1 / len(FULL))
            clear = full or abs(diff) > 2 * err
            col = ISO if full else CLEAR if clear else FAINT
            ax.errorbar(m, y, xerr=np.std(v, ddof=1), fmt='D', ms=7, color=col, mec='white', mew=0.8, lw=1.6,
                        capsize=3, zorder=4)
            ax.text(1.015, y, '—' if full else f'{diff:+.2f} ± {err:.2f}', transform=ax.get_yaxis_transform(), va='center',
                    fontsize=8.5, color=INK if clear else FAINT, fontweight='bold' if clear and not full else 'normal')
        ax.text(1.015, 0.6 + 0.35, 'Δ ± 1σ (points)', transform=ax.get_yaxis_transform(), va='bottom', fontsize=8,
                color=INK)
        ax.set_yticks(ys, [r[0] for r in rows], fontsize=9)
        ax.get_yticklabels()[0].set_fontweight('bold')
        ax.set_ylim(ys[-1] - 0.6, 0.6)
        ax.grid(False, axis='y')
        ax.set_title(title, loc='left', fontsize=9.5, color=INK)
    axes[-1].set_xlabel('Missed injections, FF < 0.97 (%)')
    axes[-1].set_xlim(2.5, 7.2)
    axes[-1].legend(handles=[
        plt.Line2D([], [], ls='-', marker='D', color=INK, ms=7, lw=1.6, label=r'$\mu \pm \sigma$')],
        loc='center right', fontsize=7.5, frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
