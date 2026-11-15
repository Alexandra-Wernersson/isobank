"""
plot_mm_sweep.py -- mbank's minimal-match sweep (eccentric 5D): coverage against bank size, with isobank.

    python figures/plot_mm_sweep.py --csv figures/mm_sweep.csv --out figures/bank_size_vs_coverage.pdf

--csv (figures/mm_sweep.csv, the numbers of iclr_plotting.ipynb) has one row per mbank bank, columns: mm, cf, templates, coverage, coverage_lowmc (coverages in %, FF >= 0.97,
10,000 injections of seed 4002, k = 500; coverage_lowmc for chirp mass < 1.4 Msun). Circles: all injections (left
axis); open squares: chirp mass < 1.4 Msun (right axis). Colour = covering fraction, grey for the paper's 0.9 (as
the paper-setting runs of plot_tuning_violins.py); isobank's paper bank in its blue, with dotted lines at its
coverage (dotted: all injections, dashed: chirp mass < 1.4 Msun).
"""
import argparse, csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

ISO, INK = '#0284c7', '#111827'
CF_COLOURS = {0.9: '#8b8f98', 0.99: '#117733', 0.6: '#aa4499'}   # the palette of plot_tuning_violins.py
LABEL_OFFSETS = {   # (mm, cf) -> (dx, dy, ha) in points, for labels that would overlap (from iclr_plotting.ipynb)
    (0.979, 0.9): (-8, 6, 'right'), (0.98, 0.9): (6, -4, 'left'), (0.977, 0.9): (-6, -12, 'right'),
    (0.978, 0.9): (6, -12, 'left'), (0.98775, 0.6): (6, 6, 'left'), (0.97, 0.99): (-8, -12, 'right'),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--isobank', default='2267558,96.64,91.3',
                    help="isobank's templates, coverage and low-Mc coverage (default: the paper's bank)")
    ap.add_argument('--out', default='figures/bank_size_vs_coverage.pdf')
    args = ap.parse_args()
    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(args.csv))]
    n_iso, c_iso, l_iso = (float(x) for x in args.isobank.split(','))

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax2 = ax.twinx()
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)   # circles above the squares
    for r in rows:
        col = CF_COLOURS.get(r['cf'], INK)
        ax.plot(r['templates'] / 1e6, r['coverage'], 'o', ms=7, color=col, mec='white', mew=0.8, zorder=3)
        ax2.plot(r['templates'] / 1e6, r['coverage_lowmc'], 's', ms=6, mfc='white', mec=col, mew=1.6, zorder=3)
        dx, dy, ha = LABEL_OFFSETS.get((r['mm'], r['cf']), (6, -12, 'left'))
        ax.annotate(rf"$\mu$ = {r['mm']:g}", (r['templates'] / 1e6, r['coverage']), xytext=(dx, dy),
                    textcoords='offset points', ha=ha, fontsize=7.5, color=INK)
    ax.plot(n_iso / 1e6, c_iso, '*', ms=15, color=ISO, mec='white', mew=0.8, zorder=4)
    ax2.plot(n_iso / 1e6, l_iso, '*', ms=15, mfc='white', mec=ISO, mew=1.4, zorder=4)
    ax.axhline(c_iso, color=ISO, ls=':', lw=1); ax.axvline(n_iso / 1e6, color=ISO, ls=':', lw=1)
    ax2.axhline(l_iso, color=ISO, ls='--', lw=1, alpha=0.6)

    ax.set_xscale('log')
    ax.set_xticks([2, 3, 5, 10, 20], ['2', '3', '5', '10', '20'])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Number of templates (millions)')
    ax.set_ylabel(r'Coverage, FF $\geq$ 0.97 (%)')
    ax2.set_ylabel(r'Coverage for $\mathcal{M}_c < 1.4\,M_\odot$ (%)')
    ax2.grid(False)
    handles = [plt.Line2D([], [], ls='', marker='*', ms=12, color=ISO, label='isobank (paper)')]
    handles += [plt.Line2D([], [], ls='', marker='o', ms=7, color=CF_COLOURS[cf], label=f'mbank, cf = {cf:g}')
                for cf in sorted({r['cf'] for r in rows}) if cf in CF_COLOURS]
    handles += [plt.Line2D([], [], ls='', marker='o', ms=7, color=INK, label='all injections'),
                plt.Line2D([], [], ls='', marker='s', ms=6, mfc='white', mec=INK, mew=1.6,
                           label=r'$\mathcal{M}_c < 1.4\,M_\odot$')]
    ax.legend(handles=handles, loc='lower right', fontsize=7.5, frameon=True, framealpha=0.95, edgecolor='#d1d5db')
    plt.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
