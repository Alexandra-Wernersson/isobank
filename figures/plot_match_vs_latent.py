"""
plot_match_vs_latent.py -- is latent distance a faithful surrogate for the match?

    python scripts/plot_match_vs_latent.py --config configs/eccentric_5d.py \
        --out figures/match_vs_latent.pdf

With --mode distance the same pairs are used to plot proper distance against
latent distance, where exact isometry is the diagonal.
"""
import argparse, importlib.util, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import load_checkpoint, metric_guided_pairs


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
    ap.add_argument('--out', default='match_vs_latent.pdf')
    ap.add_argument('--mode', choices=['match', 'distance'], default='match')
    ap.add_argument('--n-pairs', type=int, default=3000)
    ap.add_argument('--fracs', default='0.05,2.5', help='separation range in r_target units')
    ap.add_argument('--bins', type=int, default=25)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--color', default='#0284c7')
    ap.add_argument('--no-rasterize', action='store_true')
    args = ap.parse_args()

    C = load_config(args.config)
    if not os.path.exists(C.BEST_CKPT):
        raise SystemExit(
            f"no trained flow at {C.BEST_CKPT}\n"
            f"run:  python scripts/train_flow.py --config {args.config}")
    flow, ckpt = load_checkpoint(C.BEST_CKPT)
    metric = C.make_metric()
    r_target = float(ckpt['r_target'])
    f_lo, f_hi = (float(v) for v in args.fracs.split(','))

    #	anchors from the case's own sampler, then metric-guided partners
    anchors, _ = C.sample_data(n_train=args.n_pairs, n_val=8, seed=args.seed)
    g = metric.get_metric(torch.tensor(anchors), metric_type='hessian')
    valid_fn = None if C.valid_mask is None else (lambda t: bool(C.valid_mask(t[None])[0]))
    t1, t2 = metric_guided_pairs(anchors, g, r_target, args.n_pairs, (f_lo, f_hi),
                                 C.lo, C.hi, seed=args.seed, phi_col=C.phi_col,
                                 valid_fn=valid_fn)

    with torch.no_grad():
        z1, _ = flow(t1); z2, _ = flow(t2)
        d_z = torch.linalg.norm(z1 - z2, dim=1).numpy()
        y = np.asarray(metric.match(t1, t2) if args.mode == 'match'
                       else metric.proper_distance(t1, t2))

    ok = np.isfinite(y) & np.isfinite(d_z)
    d_z, y = d_z[ok], y[ok]
    edges = np.linspace(0, d_z.max(), args.bins + 1)
    cen, mu, sd = [], [], []
    for i in range(args.bins):
        sel = (d_z >= edges[i]) & (d_z < edges[i + 1])
        if sel.sum() > 5:
            cen.append(0.5 * (edges[i] + edges[i + 1]))
            mu.append(y[sel].mean()); sd.append(y[sel].std())
    cen, mu, sd = map(np.array, (cen, mu, sd))

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.scatter(d_z, y, s=5, alpha=0.22, lw=0, color=args.color, rasterized=_rasterize_ok(args.no_rasterize))
    ax.plot(cen, mu, 'o-', color=args.color, ms=4.2, lw=2.0, label='binned mean')
    ax.fill_between(cen, mu - sd, mu + sd, color=args.color, alpha=0.18)
    if args.mode == 'match':
        ax.axhline(1 - C.MU, color='k', ls=':', lw=1.0, label=f'minimal match {1-C.MU:.2f}')
        ax.set_ylabel(r'Match $\mathcal{M}(\theta_1,\theta_2)$', fontsize=11)
    else:
        lim = max(d_z.max(), y.max())
        ax.plot([0, lim], [0, lim], 'k--', lw=1.0, label='exact isometry')
        ax.set_ylabel('Proper distance', fontsize=11)
    ax.axvline(r_target, color='#dc2626', ls='--', lw=1.3,
               label=rf'$r_{{\rm target}}$ = {r_target:.3f}')
    ax.set_xlabel(r'Latent distance $\|z(\theta_1)-z(\theta_2)\|$', fontsize=11)
    ax.grid(True, alpha=0.25); ax.legend(fontsize=8.5, loc='lower left', framealpha=0.9)
    plt.tight_layout(); plt.savefig(args.out, dpi=150, bbox_inches='tight', facecolor='white')

    near = np.abs(d_z - r_target) < 0.1 * r_target
    if near.sum() > 5:
        print(f"at r_target: mean {'match' if args.mode=='match' else 'proper distance'} "
              f"= {y[near].mean():.4f} ({near.sum()} pairs)")
    print(f"{len(d_z):,} pairs -> {args.out}")


if __name__ == '__main__':
    main()
