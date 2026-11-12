"""
latent_neighbours.py -- true match of the points a latent space puts within r_target.

    python autoencoder_comparison/latent_neighbours.py --config configs/eccentric_5d.py \
        --isobank runs/eccentric_5d/flow_best.pt \
        --autoencoder autoencoder_comparison/runs/eccentric_5d/autoencoder_latent5_encoder.pt \
        --autoencoder autoencoder_comparison/runs/eccentric_5d/autoencoder_latent5_decoder.pt

Random points are mapped to the latent space, and pairs closer than r_target
there are matched with true waveforms. r_target is where the mean mismatch is
mu, so for a faithful map these pairs match well; a map that folds distant
regions onto each other gives pairs that are far apart in the parameters and
match badly, which placement then leaves uncovered. The local monitoring pairs
of train_autoencoder.py can't see this.
"""
import argparse, os, sys
import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isobank import load_checkpoint
from train_autoencoder import Autoencoder, Scaling, load_config


def latent_maps(args, C):
    maps = []
    if args.isobank:
        flow, _ = load_checkpoint(args.isobank)
        maps.append(('isobank', lambda t: flow(torch.as_tensor(t))[0]))
    scaling = Scaling(C.lo, C.hi)
    for path in args.autoencoder:
        model = Autoencoder(C.D, C.D)
        model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        name = os.path.splitext(os.path.basename(path))[0].replace('autoencoder_', 'autoencoder ')
        maps.append((name, lambda t, m=model: m.encoder(scaling.x(t))))
    return maps


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--isobank', default=None, help="isobank's flow_best.pt")
    ap.add_argument('--autoencoder', action='append', default=[], help='state dict from train_autoencoder.py')
    ap.add_argument('--r-target', type=float, default=None, help="default: the config's R_TARGET_FILE")
    ap.add_argument('--n-points', type=int, default=400_000)
    ap.add_argument('--n-pairs', type=int, default=3000)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    C = load_config(args.config)
    r = args.r_target or float(np.load(C.R_TARGET_FILE)[0])
    metric = C.make_metric()
    rng = np.random.default_rng(args.seed)
    theta = rng.uniform(C.lo, C.hi, (args.n_points, C.D)).astype(np.float32)
    if C.valid_mask is not None:
        theta = theta[C.valid_mask(theta)]
    print(f"{C.CASE_NAME}: {len(theta):,} random points, r_target = {r:.4f}\n")
    print(f"{'latent space':<32}{'pairs < r':>12}{'median match':>14}{'match < 0.97':>14}{'match < 0.5':>13}")

    for name, encode in latent_maps(args, C):
        with torch.no_grad():
            z = np.vstack([encode(theta[i:i + 100_000]).numpy() for i in range(0, len(theta), 100_000)])
        pairs = cKDTree(z).query_pairs(r, output_type='ndarray')
        if len(pairs) == 0:
            print(f"{name:<32}{0:>12}")
            continue
        pick = pairs[rng.choice(len(pairs), min(args.n_pairs, len(pairs)), replace=False)]
        with torch.no_grad():
            match = np.asarray(metric.match(torch.tensor(theta[pick[:, 0]]), torch.tensor(theta[pick[:, 1]])))
        print(f"{name:<32}{len(pairs):>12,}{np.median(match):>14.3f}{100 * np.mean(match < 0.97):>13.1f}%"
              f"{100 * np.mean(match < 0.5):>12.1f}%")


if __name__ == '__main__':
    main()
