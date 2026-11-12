"""
plot_comparison.py -- the two comparison figures, isobank vs the autoencoders.

    python autoencoder_comparison/plot_comparison.py --runs autoencoder_comparison/runs/eccentric_5d \
        --isobank-log logs/train_isobank.log --isobank-ff runs/eccentric_5d/ff_isobank.npy \
        --isobank-bank runs/eccentric_5d/bank_isobank.npy [--extra-ff ff_mbank.npy:"mbank (2.95M)"]

  * training.pdf: each method's training loss (relative to its first logged value)
    and the 10th percentile and median of d_z / d_proper on the held-out
    monitoring pairs, against the fraction of training done;
  * ff_cdf.pdf: cumulative distribution of fitting factors (figures/plot_ff_cdf.py);
  * ff_summary.md: the spread of the fitting factors of each bank (summarise_ff.py).

The autoencoder runs are the history_*.json, bank_*.npy and ff_*.npy files in --runs.
"""
import argparse, glob, json, os, re, subprocess, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = ['#0284c7', '#d97706', '#059669', '#7c3aed', '#dc2626']     # as figures/plot_ff_cdf.py
LABELS = {'latent5_encoder': 'autoencoder, encoder metric loss',
          'latent5_decoder': 'autoencoder, decoder metric loss (App. B)'}


def isobank_history(log):
    """Epoch, loss, monitor median and p10 from the log of scripts/train_flow.py."""
    rows = []
    with open(log) as fh:
        for line in fh:
            m = re.search(r'Epoch (\d+) \| loss=([-\d.e+]+).*monitor med=([\d.]+) p10=([\d.]+)', line)
            if m:
                rows.append([float(v) for v in m.groups()])
    if not rows:
        raise SystemExit(f"no training lines in {log}")
    e, loss, med, p10 = np.array(rows).T
    return dict(epoch=e, loss=loss, p50=med, p10=p10)


def autoencoder_histories(runs):
    out = {}
    for path in sorted(glob.glob(os.path.join(runs, 'history_*.json'))):
        tag = os.path.basename(path)[len('history_'):-len('.json')]
        with open(path) as fh:
            h = json.load(fh)['history']
        out[tag] = {k: np.array(v) for k, v in h.items()}
    return out


def human(n):
    return f"{n / 1e6:.2f}M" if n >= 1e6 else f"{n / 1e3:.0f}k"


def plot_training(curves, out):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for (label, h), color in zip(curves, PALETTE):
        x = h['epoch'] / h['epoch'].max()
        ax1.semilogy(x, h['loss'] / h['loss'][0], color=color, lw=1.8, label=label)
        ax2.plot(x, h['p50'], color=color, lw=1.8, label=f'{label}: median')
        ax2.plot(x, h['p10'], color=color, lw=1.4, ls=(0, (4, 2)), label=f'{label}: p10')
    ax1.set_ylabel('training loss / first value')
    ax2.axhline(1, color='#6b7280', lw=1, ls=':')
    ax2.set_ylabel(r'$d_z / d_{\rm proper}$, held-out pairs')
    ax2.set_ylim(0, 1.5)
    for ax in (ax1, ax2):
        ax.set_xlabel('fraction of training')
        ax.grid(True, which='both', alpha=0.25)
    ax1.legend(fontsize=8.5, framealpha=0.92)
    ax2.legend(fontsize=7.5, framealpha=0.92, ncol=2, loc='upper center',
               bbox_to_anchor=(0.5, -0.17))
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True, help='folder of the autoencoder runs')
    ap.add_argument('--isobank-log', required=True)
    ap.add_argument('--isobank-ff', required=True)
    ap.add_argument('--isobank-bank', required=True)
    ap.add_argument('--extra-ff', action='append', default=[], help='path:label, e.g. mbank')
    ap.add_argument('--out', default=None, help='output folder (default: --runs)')
    args = ap.parse_args()
    out = args.out or args.runs

    tags = [t for t in LABELS if os.path.exists(os.path.join(args.runs, f'ff_{t}.npy'))]
    histories = autoencoder_histories(args.runs)
    curves = [('isobank', isobank_history(args.isobank_log))]
    curves += [(LABELS[t], histories[t]) for t in tags if t in histories]
    plot_training(curves, os.path.join(out, 'training.pdf'))

    n_iso = len(np.load(args.isobank_bank, mmap_mode='r'))
    specs = [f"{args.isobank_ff}:isobank ({human(n_iso)})"]
    for t in tags:
        n = len(np.load(os.path.join(args.runs, f'bank_{t}.npy'), mmap_mode='r'))
        specs.append(f"{os.path.join(args.runs, f'ff_{t}.npy')}:{LABELS[t]} ({human(n)})")
    specs += args.extra_ff
    cmd = [sys.executable, os.path.join(ROOT, 'figures', 'plot_ff_cdf.py'),
           '--out', os.path.join(out, 'ff_cdf.pdf')]
    for s in specs:
        cmd += ['--ff', s]
    subprocess.run(cmd, check=True)

    table = subprocess.run([sys.executable, os.path.join(ROOT, 'autoencoder_comparison', 'summarise_ff.py'),
                            *specs], check=True, capture_output=True, text=True).stdout
    with open(os.path.join(out, 'ff_summary.md'), 'w') as fh:
        fh.write(table)
    print(table + f"-> {os.path.join(out, 'ff_summary.md')}")


if __name__ == '__main__':
    main()
