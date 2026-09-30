"""
plot_bank_map.py -- projections of a template bank onto pairs of parameters.

    python scripts/plot_bank_map.py --config configs/eccentric_5d.py \
        --bank runs/eccentric_5d/bank.npy --out figures/bank_map.pdf

Shows where the placement put templates: the density follows the metric volume,
so it concentrates wherever waveforms change fastest with the parameters.

NOTE: with --subsample large, the points are rasterized to keep the PDF small.
matplotlib < 3.10 writes corrupt rasterized images into PDFs (they render as
blank strips in some viewers); use matplotlib >= 3.10, or pass --no-rasterize.
"""
import argparse, importlib.util, itertools, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _rasterize_ok(disabled=False):
    """matplotlib < 3.10 writes corrupt rasterized images into PDFs; fall back
    to vector points (larger file, but valid) with a warning."""
    if disabled:
        return False
    ver = tuple(int(v) for v in matplotlib.__version__.split('.')[:2])
    if ver < (3, 10):
        print(f"warning: matplotlib {matplotlib.__version__} writes corrupt rasterized "
              f"PDF images; drawing vector points instead (upgrade to >= 3.10)")
        return False
    return True


def load_config(path):
    spec = importlib.util.spec_from_file_location('case_config', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--bank', required=True)
    ap.add_argument('--out', default='bank_map.pdf')
    ap.add_argument('--subsample', type=int, default=60_000,
                    help='templates to draw (default 60000; 0 = all)')
    ap.add_argument('--pairs', default=None,
                    help='column pairs, e.g. "0,1 0,4 2,4" (default: first six)')
    ap.add_argument('--color', default='#0284c7')
    ap.add_argument('--no-rasterize', action='store_true')
    ap.add_argument('--dpi', type=int, default=200)
    args = ap.parse_args()

    C = load_config(args.config)
    labels = getattr(C, 'PARAM_LABELS', [f'$x_{{{i}}}$' for i in range(C.D)])
    bank = np.load(args.bank)
    if args.subsample and args.subsample < len(bank):
        bank = bank[np.random.default_rng(5).permutation(len(bank))[:args.subsample]]

    pairs = ([tuple(int(v) for v in p.split(',')) for p in args.pairs.split()]
             if args.pairs else list(itertools.combinations(range(C.D), 2))[:6])
    raster = _rasterize_ok(args.no_rasterize)
    ncol = min(3, len(pairs)); nrow = int(np.ceil(len(pairs) / ncol))
    fig = plt.figure(figsize=(5 * ncol, 4.1 * nrow), facecolor='white')
    gs = gridspec.GridSpec(nrow, ncol, figure=fig, hspace=0.30, wspace=0.28)
    for k, (xi, yi) in enumerate(pairs):
        ax = fig.add_subplot(gs[k // ncol, k % ncol])
        ax.scatter(bank[:, xi], bank[:, yi], s=2.5, alpha=0.3, lw=0,
                   color=args.color, rasterized=raster)
        ax.set_xlabel(labels[xi], fontsize=16); ax.set_ylabel(labels[yi], fontsize=16)
        ax.tick_params(axis='both', labelsize=12); ax.grid(True, alpha=0.25)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches='tight', facecolor='white')
    print(f"{len(bank):,} templates drawn -> {args.out}")


if __name__ == '__main__':
    main()
