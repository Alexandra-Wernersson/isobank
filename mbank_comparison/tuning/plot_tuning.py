"""
plot_tuning.py -- the figures and table of the mbank tuning sweep.

    python plot_tuning.py --work WORK --variants variants.tsv [--isobank-dir ISOBANK/runs/eccentric_5d]

  * tuning.pdf: bank size against coverage (FF >= 0.97) for each variant, as
    figures/bank_size_vs_coverage.png, with isobank and the paper's mbank bank;
  * tuning_loss.pdf: the validation loss of each trained flow;
  * tuning.csv: per variant, size, coverage, the spread of the fitting factors and the flow diagnostics.

WORK/<variant>/ holds bank.npy, ff.npy, diagnostics.json and history_*.npz.
"""
import argparse, csv, glob, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

THRESHOLD = 0.97
PAPER = dict(isobank=(2_267_558, 96.64, 0.18), mbank=(2_953_739, 94.17, 0.23))   # mbank_comparison/README.md
GROUPS = [('paper', 'paper settings', '#6b7280'), ('weighted', 'loss: weighted_ll_mse', '#0284c7'),
          ('quantile', 'C start quantile', '#d97706'), ('padding', 'boundary padding', '#059669'),
          ('lr', 'optimiser', '#7c3aed'), ('batch', 'optimiser', '#7c3aed'),
          ('data', 'dataset size', '#0891b2'), ('is', 'importance sampling', '#be185d')]


def read_variants(path):
    with open(path) as fh:
        rows = [l.rstrip('\n').split('\t') for l in fh if l.strip() and not l.startswith('#')]
    return [dict(name=r[0], dataset=r[1], flow=r[2]) for r in rows]


def group(name):
    for prefix, label, color in GROUPS:
        if name.startswith(prefix):
            return label, color
    return 'other', '#374151'


def coverage(ff):
    p = (ff >= THRESHOLD).mean()
    return 100 * p, 100 * np.sqrt(p * (1 - p) / len(ff))


def collect(work, variants):
    rows = []
    for v in variants:
        d = os.path.join(work, v['name'])
        bank, ff = os.path.join(d, 'bank.npy'), os.path.join(d, 'ff.npy')
        if not (os.path.exists(bank) and os.path.exists(ff)):
            print(f"{v['name']}: no bank or fitting factors yet, skipped")
            continue
        f = np.load(ff)
        cov, err = coverage(f)
        p1, p5 = np.percentile(f, [1, 5])
        row = dict(name=v['name'], templates=len(np.load(bank, mmap_mode='r')), coverage=cov, error=err,
                   ff_mean=f.mean(), ff_std=f.std(ddof=1), ff_p1=p1, ff_p5=p5)
        diag = os.path.join(d, 'diagnostics.json')
        if os.path.exists(diag):
            with open(diag) as fh:
                row.update(json.load(fh))
        rows.append(row)
    return rows


def plot_sizes(rows, isobank, out):
    fig, ax = plt.subplots(figsize=(9.5, 5))
    seen = set()
    for r in rows:
        label, color = group(r['name'])
        ax.errorbar(r['templates'] / 1e6, r['coverage'], yerr=r['error'], fmt='o', ms=7, color=color,
                    capsize=2, label=None if label in seen else f'mbank, {label}')
        seen.add(label)
        ax.annotate(r['name'], (r['templates'] / 1e6, r['coverage']), xytext=(5, 4),
                    textcoords='offset points', fontsize=8, color='#374151')
    n, cov, err = PAPER['mbank']
    ax.errorbar(n / 1e6, cov, yerr=err, fmt='s', mfc='none', ms=8, color='#6b7280', capsize=2,
                label='mbank, paper bank')
    n, cov, err, label = isobank
    ax.errorbar(n / 1e6, cov, yerr=err, fmt='*', ms=16, color='#dc2626', capsize=2, label=label)
    ax.set_xscale('log')
    for axis in (ax.xaxis.set_major_formatter, ax.xaxis.set_minor_formatter):
        axis(FormatStrFormatter('%g'))
    ax.set_xlabel('Number of templates (millions)')
    ax.set_ylabel(f'Coverage, FF ≥ {THRESHOLD} (%)')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8.5, framealpha=0.92, loc='upper left', bbox_to_anchor=(1.01, 1))
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"-> {out}")


def plot_losses(work, variants, out):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for v in variants:
        files = glob.glob(os.path.join(work, v['name'], 'history_*.npz'))
        if v['flow'] != 'train' or not files:
            continue
        with np.load(files[0], allow_pickle=False) as h:
            loss, step = h['validation_loss'], int(h['validation_step'])
        label, color = group(v['name'])
        ax.semilogy(step * np.arange(len(loss)), loss, color=color, lw=1.4, label=v['name'])
    ax.set_xlabel('epoch')
    ax.set_ylabel('validation loss')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=7.5, ncol=2, framealpha=0.92)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', required=True)
    ap.add_argument('--variants', required=True)
    ap.add_argument('--isobank-dir', default=None,
                    help='with bank_isobank.npy and ff_isobank.npy; else the paper values')
    args = ap.parse_args()

    variants = read_variants(args.variants)
    rows = collect(args.work, variants)

    iso_bank = args.isobank_dir and os.path.join(args.isobank_dir, 'bank_isobank.npy')
    iso_ff = args.isobank_dir and os.path.join(args.isobank_dir, 'ff_isobank.npy')
    if iso_bank and os.path.exists(iso_bank) and os.path.exists(iso_ff):
        isobank = (len(np.load(iso_bank, mmap_mode='r')), *coverage(np.load(iso_ff)), 'isobank')
    else:
        isobank = (*PAPER['isobank'], 'isobank (paper bank)')

    plot_sizes(rows, isobank, os.path.join(args.work, 'tuning.pdf'))
    plot_losses(args.work, variants, os.path.join(args.work, 'tuning_loss.pdf'))

    keys = ['name', 'templates', 'coverage', 'error', 'ff_mean', 'ff_std', 'ff_p1', 'ff_p5', 'abs_r_p50', 'abs_r_p90', 'frac_weight_above_10',
            'C_minus_Cbest', 'samples_inside', 'log10_volume', 'log10_volume_inside']
    with open(os.path.join(args.work, 'tuning.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'variant':<15}{'templates':>12}{'coverage %':>13}{'|r| p50':>9}{'inside':>8}")
    for r in rows:
        print(f"{r['name']:<15}{r['templates']:>12,}{r['coverage']:>8.2f} ± {r['error']:.2f}"
              f"{r.get('abs_r_p50', np.nan):>9.3f}{r.get('samples_inside', np.nan):>8.3f}")


if __name__ == '__main__':
    main()
