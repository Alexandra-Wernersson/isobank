"""
generate_bank.py -- place a template bank in the flow's latent space.

    python scripts/generate_bank.py --config configs/eccentric_5d.py \
        --radius-factor 1.0 --out runs/eccentric_5d/bank.npy

Candidates are drawn from the parameter space, mapped through the flow, and
accepted when their nearest existing template is farther than the placement
radius in latent space; no metric or waveform is evaluated during placement.

The placement radius is ``--radius-factor`` times r_target. A factor of 1 runs
at the calibrated radius and over covers (the typical signal then sits well
inside the radius); larger factors give smaller banks whose typical
signal-to-template distance is near r_target. The factor therefore trades bank
size against coverage, and the resulting coverage must be measured with
scripts/evaluate_bank.py.

Placement stops when the estimated uncovered fraction of a batch falls below
--stop-acceptance, measured either as metric volume (``--stop-on volume``,
weighting candidates by the flow Jacobian) or as a plain candidate count
(``--stop-on coord``).
"""
import argparse, importlib.util, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import load_checkpoint, greedy_placement


def load_config(path):
    spec = importlib.util.spec_from_file_location('case_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='output .npy (default: <OUT_DIR>/bank.npy)')
    ap.add_argument('--radius-factor', type=float, default=1.0,
                    help='placement radius in units of r_target (default 1.0)')
    ap.add_argument('--stop-acceptance', type=float, default=None,
                    help='stopping tolerance (default: config STOP_ACCEPTANCE)')
    ap.add_argument('--stop-on', choices=['volume', 'coord'], default='coord',
                    help="uncovered metric volume, or plain candidate acceptance (default)")
    ap.add_argument('--batch', type=int, default=50_000)
    ap.add_argument('--max-batches', type=int, default=5_000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    C = load_config(args.config)
    out = args.out or os.path.join(C.OUT_DIR, 'bank.npy')
    partial = out.replace('.npy', '_partial.npy')

    if not os.path.exists(C.BEST_CKPT):
        raise SystemExit(
            f"no trained flow at {C.BEST_CKPT}\n"
            f"run:  python scripts/train_flow.py --config {args.config}")
    flow, ckpt = load_checkpoint(C.BEST_CKPT)
    r_target = float(ckpt['r_target'])
    r_place = args.radius_factor * r_target
    stop_acc = args.stop_acceptance if args.stop_acceptance is not None \
        else getattr(C, 'STOP_ACCEPTANCE', 0.01)

    print(f"{C.CASE_NAME}\nr_target = {r_target:.5f}   radius factor = {args.radius_factor}"
          f"   r_place = {r_place:.5f}\nstop: {args.stop_on} acceptance < {stop_acc}")
    sys.stdout.flush()

    initial = np.load(partial) if os.path.exists(partial) else None
    if initial is not None:
        print(f"resuming from partial bank: {len(initial):,} templates")

    t0 = time.time()
    templates = greedy_placement(flow, r_place, C.lo, C.hi, embed_fn=C.embed_fn,
                                 valid_fn=C.valid_mask, batch=args.batch,
                                 max_batches=args.max_batches,
                                 stop_acceptance=stop_acc, stop_on=args.stop_on,
                                 seed=args.seed, initial_templates=initial,
                                 partial_save=partial, verbose=True)
    np.save(out, templates)
    print(f"\nPlaced {len(templates):,} templates in {(time.time()-t0)/60:.1f} min -> {out}")


if __name__ == '__main__':
    main()
