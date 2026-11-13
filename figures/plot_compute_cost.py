"""
plot_compute_cost.py -- CPU-hours (wall time x CPUs) to build each bank, per step, eccentric 5D at Nikhef.

    python figures/plot_compute_cost.py --out figures/compute_cost.pdf

The numbers are RemoteWallClockTime x RequestCpus from condor_history of the Nikhef DAGs (summed over jobs).
isobank's train_isobank node computes the radius and the training pairs before training, so its preprocessing is
part of training. mbank: the three runs with the paper's settings (mean, bar = min-max). Cabass et al.: 20 sub-banks,
latent 3; 'placement' is counting the lattice cells (count20), and preprocessing includes the waveforms, the SVD and
the bank-size regressor. Scoring against the injections (5.7 CPU-h, the same for every method) is left out.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#111827'
STEPS = [('preprocessing', '^'), ('training', 's'), ('placement', 'D'), ('total', 'o')]
MODELS = [   # name, colour (as in the FF figures), {step: [CPU-h per run]}
    ('isobank', '#0284c7', dict(training=[3.35], placement=[0.35])),
    ('mbank', '#8b8f98', dict(preprocessing=[0.12] * 3, training=[0.19, 0.12, 0.18], placement=[1.23, 0.77, 1.56])),
    ('Cabass et al.\n(2025)', '#059669', dict(preprocessing=[95.8], training=[21.2], placement=[7.2])),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='figures/compute_cost.pdf')
    args = ap.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    offs = np.linspace(-0.27, 0.27, len(STEPS))
    for xi, (name, col, steps) in enumerate(MODELS):
        steps = dict(steps, total=list(np.sum([steps[s] for s in steps], axis=0)))
        for (step, marker), dx in zip(STEPS, offs):
            if step not in steps:
                ax.text(xi + dx, 0.12, 'in\ntraining', ha='center', va='center', fontsize=7, color=INK, alpha=0.6)
                continue
            v = np.array(steps[step]); m = v.mean()
            big = step == 'total'
            ax.errorbar(xi + dx, m, yerr=[[m - v.min()], [v.max() - m]] if len(v) > 1 else None, fmt=marker,
                        ms=10 if big else 7.5, color=col, mfc=col if big else 'white', mew=1.8, lw=1.4, capsize=0,
                        zorder=3)
            lab = f'{m:.2f}' if m < 1 else f'{m:.1f}' if m < 10 else f'{m:.0f}'
            ax.annotate(lab, (xi + dx, v.max()), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=7.5, color=INK, fontweight='bold' if big else 'normal')
    ax.set_yscale('log'); ax.set_ylim(0.05, 400)
    ax.set_yticks([0.1, 1, 10, 100], ['0.1', '1', '10', '100'])
    ax.set_ylabel('Wall time on 1 CPU (CPU-hours)')
    ax.set_xticks(range(len(MODELS)), [m[0] for m in MODELS], fontsize=10)
    ax.get_xticklabels()[0].set_fontweight('bold')
    ax.set_xlim(-0.55, len(MODELS) - 0.45); ax.grid(False, axis='x')
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=mk, ms=10 if s == 'total' else 7.5, color=INK,
                                  mfc=INK if s == 'total' else 'white', mew=1.8, label=s) for s, mk in STEPS],
              loc='upper left', ncol=4, fontsize=8.5, frameon=False, handletextpad=0.3, columnspacing=1.2)
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
