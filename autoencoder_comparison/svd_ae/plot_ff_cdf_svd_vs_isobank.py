"""
plot_ff_cdf_svd_vs_isobank.py -- FF cumulative distributions, isobank vs Cabass et al. (2025), arXiv:2511.10466
(SVD coefficients -> autoencoder -> lattice, with 10 and 20 sub-banks), eccentric 5D, in the style of
mbank_comparison/figure/ff_cdf_isobank_vs_mbank.py, with the plain SVD grid underneath (Alexandra's baseline).
Like Cabass et al.'s, the SVD grid's templates are waveforms rebuilt from SVD coefficients, not exact waveforms.

    python autoencoder_comparison/svd_ae/plot_ff_cdf_svd_vs_isobank.py --runs autoencoder_comparison/runs/eccentric_5d/svd_ae

All curves: the first 4,000 injections of seed 4002 (Alexandra's SVD tests). Each Cabass et al. lattice at the
spacing whose bank size is closest to isobank's (as svd_ae.py evaluate); both SVD methods are scored with the
phase overlap of
svd_grid_refined.py, without an extra time/phase search; isobank uses the true match.
Reads the ff_*.npy files and lattice_table_sb20.md of --runs; writes ff_cdf_svd_vs_isobank.pdf there.
"""
import argparse, os, re
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

THR = 0.97
CURVES = [   # file in --runs, lattice_table row (label, spacing) or None, colour, line style, label
    ('ff_isobank_first4000.npy', None, '#0284c7', '-', 'isobank'),
    ('ff_svd_a0.44.npy', ('SVD, BCC on c0..c2', '0.44'), '#d97706', '-', 'SVD grid, no autoencoder'),
    ('ff_ae3sb20_a0.48.npy', ('SVD + autoencoder, latent 3, 20 sub-banks, BCC', '0.48'), '#059669', (0, (5, 2)),
     'Cabass et al. (2025), 20 sub-banks'),
    ('ff_ae3sb10_a0.44.npy', ('SVD + autoencoder, latent 3, 10 sub-banks, BCC', '0.44'), '#7c3aed', (0, (5, 2)),
     'Cabass et al. (2025), 10 sub-banks'),
]


def bank_sizes(table):
    """(label, spacing) -> templates, from lattice_table*.md; and isobank's size from its first line."""
    sizes, text = {}, open(table).read()
    for line in text.splitlines():
        m = re.match(r'\| (.+?) \| ([\d.]+) \| ([\d,]+) \(', line)
        if m:
            sizes[(m.group(1), m.group(2))] = int(m.group(3).replace(',', ''))
    sizes[None] = int(re.search(r'isobank: ([\d,]+) templates', text).group(1).replace(',', ''))
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', required=True)
    ap.add_argument('--table', default='lattice_table_sb20.md')
    ap.add_argument('--out', default=None, help='default: ff_cdf_svd_vs_isobank.pdf in --runs')
    args = ap.parse_args()
    sizes = bank_sizes(os.path.join(args.runs, args.table))

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    rows = []
    for name, key, col, ls, label in CURVES:
        ff = np.load(os.path.join(args.runs, name)); n = sizes[key]
        s = np.sort(ff); y = 100 * np.arange(1, len(s) + 1) / len(s)
        cov = 100 * (ff >= THR).mean(); se = 100 * np.sqrt((cov / 100) * (1 - cov / 100) / len(ff))
        ax.semilogy(s, y, color=col, ls=ls, lw=2.0, label=f'{label} ({n / 1e6:.2f}M) — {cov:.2f}%',
                    solid_capstyle='round', dash_capstyle='round')
        rows.append((label, n, cov, se, np.percentile(ff, 1), ff.min(), len(ff)))
    ax.axvline(THR, color='#dc2626', ls=':', lw=1.6, zorder=5)
    ax.set_xlabel('Fitting factor', fontsize=11)
    ax.set_ylabel('Cumulative fraction of injections (%)', fontsize=11)
    ax.set_xlim(0.85, 1.0); ax.set_ylim(0.02, 150)
    ax.tick_params(axis='both', labelsize=9)
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(fontsize=8.2, loc='upper left', framealpha=0.92)
    plt.tight_layout()
    out = args.out or os.path.join(args.runs, 'ff_cdf_svd_vs_isobank.pdf')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'saved {out}\n')
    print(f"{'bank':>34} | {'templates':>10} | {'coverage':>16} | {'p1 FF':>7} | {'min FF':>7} | N_inj")
    for label, n, cov, se, p1, mn, N in rows:
        print(f"{label:>34} | {n:>10,} | {cov:6.2f}% +/- {se:.2f} | {p1:7.4f} | {mn:7.4f} | {N:,}")


if __name__ == '__main__':
    main()
