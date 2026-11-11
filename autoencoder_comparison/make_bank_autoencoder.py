"""
make_bank_autoencoder.py -- place a bank in the latent space of a trained autoencoder.

    python autoencoder_comparison/make_bank_autoencoder.py --config configs/eccentric_5d.py \
        --model autoencoder_comparison/runs/eccentric_5d/autoencoder_latent5_encoder.pt \
        --radius-factor 1.0 --out autoencoder_comparison/runs/eccentric_5d/bank_latent5_encoder.npy

The placement is isobank's (scripts/generate_bank.py) with the same options and
defaults; the encoder of the autoencoder takes the place of the flow. r_target
is the one cached by train_autoencoder.py (the same calibration as isobank's).
"""
import argparse, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import greedy_placement
from train_autoencoder import Autoencoder, Scaling, load_config


class EncoderAsFlow(torch.nn.Module):
    """theta -> (latent, 0), the interface greedy_placement expects of a flow."""

    def __init__(self, model, scaling):
        super().__init__()
        self.model, self.scaling = model, scaling

    def forward(self, theta):
        z = self.model.encoder(self.scaling.x(theta))
        return z, torch.zeros(len(z))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True, help='state dict from train_autoencoder.py')
    ap.add_argument('--out', required=True)
    ap.add_argument('--latent-dim', type=int, default=None, help='default: the dimension of the space')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--radius-factor', type=float, default=1.0)
    ap.add_argument('--stop-acceptance', type=float, default=None)
    ap.add_argument('--stop-on', choices=['volume', 'coord'], default='coord')
    ap.add_argument('--batch', type=int, default=50_000)
    ap.add_argument('--max-batches', type=int, default=5_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    if args.stop_on == 'volume':
        raise SystemExit("--stop-on volume needs a Jacobian determinant: use coord")

    C = load_config(args.config)
    with np.load(os.path.join(os.path.dirname(args.model), 'dataset.npz'), allow_pickle=False) as f:
        r_target = float(f['r_target'])
    model = Autoencoder(C.D, args.latent_dim or C.D, args.hidden, args.layers)
    model.load_state_dict(torch.load(args.model, weights_only=True))
    model.eval()
    flow = EncoderAsFlow(model, Scaling(C.lo, C.hi))

    r_place = args.radius_factor * r_target
    stop_acc = args.stop_acceptance if args.stop_acceptance is not None \
        else getattr(C, 'STOP_ACCEPTANCE', 0.01)
    partial = args.out.replace('.npy', '_partial.npy')
    initial = np.load(partial) if os.path.exists(partial) else None
    print(f"{C.CASE_NAME}, {args.model}\nr_target = {r_target:.5f}   radius factor = "
          f"{args.radius_factor}   r_place = {r_place:.5f}\nstop: coord acceptance < {stop_acc}")
    sys.stdout.flush()

    t0 = time.time()
    with torch.no_grad():
        templates = greedy_placement(flow, r_place, C.lo, C.hi, embed_fn=C.embed_fn,
                                     valid_fn=C.valid_mask, batch=args.batch,
                                     max_batches=args.max_batches, stop_acceptance=stop_acc,
                                     stop_on=args.stop_on, seed=args.seed,
                                     initial_templates=initial, partial_save=partial,
                                     verbose=True)
    np.save(args.out, templates)
    print(f"\nPlaced {len(templates):,} templates in {(time.time() - t0) / 60:.1f} min -> {args.out}")


if __name__ == '__main__':
    main()
