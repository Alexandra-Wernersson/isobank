"""
plot_compute_cost.py -- CPU-hours (wall time x CPUs) to construct each bank, per step, and to search one hour of
data with it; eccentric 5D at Nikhef.

    python figures/plot_compute_cost.py --out figures/compute_cost.pdf

The construction numbers are RemoteWallClockTime x RequestCpus from condor_history of the Nikhef DAGs (summed over
jobs). Templates and coverage: isobank's paper bank (2.27M); its construction times are the Nikhef retrain's (3.01M), as
the paper's were not timed. isobank's train_isobank node computes the radius and the training pairs before training, so its
preprocessing is part of training. mbank: the three runs with the paper's settings (mean, bar = min-max). Cabass
et al.: 20 sub-banks, latent 3; 'placement' is counting the lattice cells (count20), and preprocessing includes
the waveforms, the SVD and the bank-size regressor. Scoring against the injections (5.7 CPU-h, the same for every
method) is left out.
'search' is matched-filtering one hour of data with the bank (these Nikhef banks, not the paper's), in PyCBC's convention (Usman et al.
2016, arXiv:1508.02357, Sec. 3.3-3.4): 256 s blocks at 4096 Hz (N = 2^20), one inverse complex FFT of 5 N log2 N
flop per template and block, 128 s of each block kept (pycbc_inspiral's start/end padding), at FLOPS per core.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INK = '#111827'
N_FFT, T_KEPT, FLOPS = 2 ** 20, 128., 2e9
SEARCH = 5 * N_FFT * np.log2(N_FFT) / T_KEPT / FLOPS   # CPU-s per template per s of data = CPU-h per template per h
STEPS = [('preprocessing', '^'), ('training', 's'), ('placement', 'D'), ('construction', 'o'), ('search', '*')]
MODELS = [   # name, colour (as in the FF figures), {step: [CPU-h per run]}, [templates per run], coverage
    ('isobank', '#0284c7', dict(training=[3.35], placement=[0.35]), [2267558], '96.6%'),   # the paper's bank
    ('mbank', '#8b8f98', dict(preprocessing=[0.12] * 3, training=[0.19, 0.12, 0.18], placement=[1.23, 0.77, 1.56]),
     [3091551, 2837943, 3005165], '93.5–94.8%'),
    ('Cabass et al. (2025)\n20 sub-banks', 'mediumseagreen', dict(preprocessing=[95.8], training=[21.2],
                                                                  placement=[7.2]), [2676002], '96.6%'),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='figures/compute_cost.pdf')
    args = ap.parse_args()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    offs = np.linspace(-0.32, 0.32, len(STEPS))
    full = lambda step: step in ('construction', 'search')
    for xi, (name, col, steps, n, _) in enumerate(MODELS):
        steps = dict(steps, construction=list(np.sum([steps[s] for s in steps], axis=0)))
        steps['search'] = list(SEARCH * np.array(n))   # one hour of data, in CPU-h
        for (step, marker), dx in zip(STEPS, offs):
            if step not in steps:
                continue
            v = np.array(steps[step]); m = v.mean()
            ax.errorbar(xi + dx, m, yerr=[[m - v.min()], [v.max() - m]] if len(v) > 1 else None, fmt=marker,
                        ms=(14 if marker == '*' else 10) if full(step) else 7.5, color=col,
                        mfc=col if full(step) else 'white', mew=1.8, lw=1.4, capsize=0, zorder=3)
            lab = f'{m:.2f}' if m < 1 else f'{m:.1f}' if m < 10 else f'{m:.0f}'
            ax.annotate(lab, (xi + dx, v.max()), xytext=(0, 7), textcoords='offset points', ha='center',
                        fontsize=7.5, color=INK, fontweight='bold' if full(step) else 'normal')
    ax.set_yscale('log'); ax.set_ylim(0.05, 4000)
    ax.set_yticks([0.1, 1, 10, 100, 1000], ['0.1', '1', '10', '100', '1000'])
    ax.set_ylabel('Wall time (CPU h)')
    ax.set_xticks(range(len(MODELS)), [f'{m[0]}\ncoverage {m[4]}' for m in MODELS], fontsize=9.5)
    ax.get_xticklabels()[0].set_fontweight('bold')
    ax.set_xlim(-0.55, len(MODELS) - 0.45); ax.grid(False, axis='x')
    ax.legend(handles=[plt.Line2D([], [], ls='', marker=mk, ms=(14 if mk == '*' else 10) if full(s) else 7.5,
                                  color=INK, mfc=INK if full(s) else 'white', mew=1.8, label=s) for s, mk in STEPS],
              loc='lower right', ncol=2, fontsize=7, frameon=True, framealpha=0.95, edgecolor='#d1d5db',
              markerscale=0.65, handletextpad=0.3, columnspacing=0.9, borderpad=0.4, labelspacing=0.35)
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}; search: {SEARCH:.2e} CPU-s per template per s of data')


if __name__ == '__main__':
    main()
