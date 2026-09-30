"""
train_flow.py -- calibrate the placement radius and train the isometric flow.

    python scripts/train_flow.py --config configs/eccentric_5d.py

Step 1 calibrates r_target: the proper distance at which the mean true waveform
mismatch equals mu (the target mismatch). It is skipped if r_target already
exists, unless --recalibrate is given.

Step 2 generates the training pairs (long-range, metric-guided local, boundary
and, if the config defines a hard sub-region, edge pairs) and trains the flow.

Outputs (paths come from the config): r_target.npy, flow_best.pt, flow_final.pt.
"""
import argparse, importlib.util, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import (calibrate_r_target, long_range_pairs, metric_guided_pairs,
                     filter_by_distance, train_isometric_flow)


def load_config(path):
    spec = importlib.util.spec_from_file_location('case_config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='case config, e.g. configs/eccentric_5d.py')
    ap.add_argument('--recalibrate', action='store_true', help='recompute r_target even if it exists')
    ap.add_argument('--epochs', type=int, default=None, help='override the config N_EPOCHS')
    args = ap.parse_args()

    C = load_config(args.config)
    metric = C.make_metric()
    valid_fn = None if C.valid_mask is None else (lambda t: bool(C.valid_mask(t[None])[0]))
    print(f"{C.CASE_NAME}"); sys.stdout.flush()

    #	--- step 1: placement radius ------------------------------------------
    if args.recalibrate or not os.path.exists(C.R_TARGET_FILE):
        print("\nCalibrating r_target against true matches..."); sys.stdout.flush()
        r_target, _ = calibrate_r_target(metric, C.lo, C.hi, mu=C.MU,
                                         n_ref=C.N_REF, valid_fn=valid_fn)
        np.save(C.R_TARGET_FILE, np.array([r_target]))
    r_target = float(np.load(C.R_TARGET_FILE)[0])
    print(f"r_target = {r_target:.5f}  (mu = {C.MU})"); sys.stdout.flush()

    #	--- step 2: training pairs --------------------------------------------
    train_data, val_data = C.sample_data()
    TAIL_W       = getattr(C, 'TAIL_W', 5.0)
    DEPTH        = getattr(C, 'DEPTH', 3)
    LOCAL_DMAX_R = getattr(C, 'LOCAL_DMAX_R', 2.0)
    LOSS_WEIGHTS = getattr(C, 'LOSS_WEIGHTS', (1.0, 5.0, 10.0, 10.0))

    print("\nLong-range pairs..."); sys.stdout.flush()
    long_pairs = long_range_pairs(train_data, metric, C.N_LONG, d_max=C.LONG_DMAX,
                                  seed=42, phi_col=C.phi_col)
    print(f"  {len(long_pairs[2])} pairs  "
          f"d=[{long_pairs[2].min():.3f}, {long_pairs[2].max():.3f}]"); sys.stdout.flush()

    g_train = metric.get_metric(torch.tensor(train_data), metric_type='hessian')
    g_val   = metric.get_metric(torch.tensor(val_data),   metric_type='hessian')

    print("\nLocal pairs..."); sys.stdout.flush()
    lt1, lt2 = metric_guided_pairs(train_data, g_train, r_target, C.N_LOCAL,
                                   C.LOCAL_FRACS, C.lo, C.hi, seed=123,
                                   phi_col=C.phi_col, valid_fn=valid_fn)
    local_pairs = filter_by_distance(metric, lt1, lt2, C.LOCAL_DMIN_R * r_target,
                                     (LOCAL_DMAX_R * r_target) if LOCAL_DMAX_R else np.inf)
    print(f"  {len(local_pairs[2])} pairs after filter"); sys.stdout.flush()

    print("\nBoundary pairs..."); sys.stdout.flush()
    bt1, bt2 = metric_guided_pairs(train_data, g_train, r_target, C.N_BOUND,
                                   C.BOUND_FRACS, C.lo, C.hi, seed=555,
                                   phi_col=C.phi_col, valid_fn=valid_fn)
    bound_pairs = filter_by_distance(metric, bt1, bt2, 0.5 * r_target, 1.5 * r_target)
    print(f"  {len(bound_pairs[2])} pairs after filter"); sys.stdout.flush()

    edge_pairs = None
    if getattr(C, 'sample_edge_anchors', None) is not None:
        print("\nEdge pairs (hard sub-region anchors)..."); sys.stdout.flush()
        e_anchors = C.sample_edge_anchors(getattr(C, 'N_EDGE_ANCHORS', 1500), seed=777)
        g_edge = metric.get_metric(torch.tensor(e_anchors), metric_type='hessian')
        et1, et2 = metric_guided_pairs(e_anchors, g_edge, r_target,
                                       getattr(C, 'N_EDGE', 5000), C.BOUND_FRACS,
                                       C.lo, C.hi, seed=778, phi_col=C.phi_col,
                                       valid_fn=valid_fn)
        edge_pairs = filter_by_distance(metric, et1, et2, 0.5 * r_target, 1.5 * r_target)
        print(f"  {len(edge_pairs[2])} pairs after filter"); sys.stdout.flush()

    print("\nMonitoring pairs (held out)..."); sys.stdout.flush()
    mt1, mt2 = metric_guided_pairs(val_data, g_val, r_target, C.N_MON,
                                   C.BOUND_FRACS, C.lo, C.hi, seed=999,
                                   phi_col=C.phi_col, valid_fn=valid_fn)
    monitor_pairs = filter_by_distance(metric, mt1, mt2, 0.5 * r_target, 1.5 * r_target)
    print(f"  {len(monitor_pairs[2])} pairs after filter"); sys.stdout.flush()

    #	--- step 3: train ------------------------------------------------------
    flow, history = train_isometric_flow(
        metric, r_target, C.variable_format,
        long_pairs, local_pairs, bound_pairs, monitor_pairs,
        d_flow=C.d_flow, n_layers=C.n_layers, hidden_dim=C.hidden_dim,
        masks=C.masks, depth=DEPTH, edge_pairs=edge_pairs,
        weights=LOSS_WEIGHTS, tail_w=TAIL_W, embed_fn=C.embed_fn,
        train_data=train_data, phi_col=C.phi_col,
        n_epochs=args.epochs if args.epochs else C.N_EPOCHS,
        best_path=C.BEST_CKPT, final_path=C.FINAL_CKPT,
        extra_ckpt={'embedded': C.embed_fn is not None})
    print(f"\nSaved: {C.BEST_CKPT}")


if __name__ == '__main__':
    main()
