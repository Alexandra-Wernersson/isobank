"""
train_flow_ablation.py -- isobank's flow with its multi-scale loss swapped out.

    python autoencoder_comparison/train_flow_ablation.py --config configs/eccentric_5d.py --loss full
    python autoencoder_comparison/train_flow_ablation.py --config configs/eccentric_5d.py --loss weights --weights 0 5 0 0
    python autoencoder_comparison/train_flow_ablation.py --config configs/eccentric_5d.py --loss jacobian

The flow (architecture, masks, optimiser, schedule, checkpoint rule) and the
pairs are those of scripts/train_flow.py; only the loss changes:

  * full: scripts/train_flow.py as it is (the reference);
  * weights: the same terms, with weights (w_global, w_local, w_bound, w_edge),
    e.g. 0 5 0 0 keeps only the local pairs;
  * jacobian: the local metric loss of the metric-preserving autoencoders,
    the mean Frobenius norm of J^T J - M at the training points, where J is
    the flow's Jacobian and M the metric.

All runs are scored on the same monitoring pairs as train_autoencoder.py
(percentiles of d_z / d_proper), whose cached data they reuse.
"""
import argparse, json, os, sys, time
import numpy as np
import torch
from torch import optim
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import long_range_pairs, metric_guided_pairs, filter_by_distance, train_isometric_flow
from isobank.flow.flow import IsometricFlow, load_checkpoint
from train_autoencoder import load_config, make_dataset


def make_pairs(C, data, out):
    """The training pairs of scripts/train_flow.py (same seeds), cached."""
    cache = os.path.join(out, 'flow_pairs.npz')
    if os.path.exists(cache):
        with np.load(cache, allow_pickle=False) as f:
            p = {k: torch.as_tensor(f[k]) for k in f.files}
        return {s: (p[s + '1'], p[s + '2'], p[s + 'd']) if s + '1' in p else None
                for s in ('long', 'local', 'bound', 'edge')}

    metric = C.make_metric()
    valid_fn = None if C.valid_mask is None else (lambda t: bool(C.valid_mask(t[None])[0]))
    r = float(data['r_target'])
    train, g_train = data['train'], data['g_train']
    local_dmax = getattr(C, 'LOCAL_DMAX_R', 2.0)          # as scripts/train_flow.py

    print("training pairs (as scripts/train_flow.py)..."); sys.stdout.flush()
    pairs = dict(long=long_range_pairs(train, metric, C.N_LONG, d_max=C.LONG_DMAX, seed=42,
                                       phi_col=C.phi_col))
    t1, t2 = metric_guided_pairs(train, g_train, r, C.N_LOCAL, C.LOCAL_FRACS, C.lo, C.hi,
                                 seed=123, phi_col=C.phi_col, valid_fn=valid_fn)
    pairs['local'] = filter_by_distance(metric, t1, t2, C.LOCAL_DMIN_R * r,
                                        local_dmax * r if local_dmax else np.inf)
    t1, t2 = metric_guided_pairs(train, g_train, r, C.N_BOUND, C.BOUND_FRACS, C.lo, C.hi,
                                 seed=555, phi_col=C.phi_col, valid_fn=valid_fn)
    pairs['bound'] = filter_by_distance(metric, t1, t2, 0.5 * r, 1.5 * r)
    pairs['edge'] = None
    if hasattr(C, 'sample_edge_anchors'):
        anchors = C.sample_edge_anchors(getattr(C, 'N_EDGE_ANCHORS', 1500), seed=777)
        g_edge = metric.get_metric(torch.tensor(anchors), metric_type='hessian')
        t1, t2 = metric_guided_pairs(anchors, g_edge, r, getattr(C, 'N_EDGE', 5000),
                                     C.BOUND_FRACS, C.lo, C.hi, seed=778, phi_col=C.phi_col,
                                     valid_fn=valid_fn)
        pairs['edge'] = filter_by_distance(metric, t1, t2, 0.5 * r, 1.5 * r)

    flat = {f'{s}{i}': np.asarray(v[j]) for s, v in pairs.items() if v is not None
            for j, i in enumerate('12d')}
    np.savez(cache, **flat)
    return pairs


def monitor_pairs(data):
    return tuple(torch.as_tensor(data[k], dtype=torch.float32) for k in ('mon1', 'mon2', 'mon_d'))


def score(flow, data):
    """Percentiles (10, 50, 90) of d_z / d_proper on the monitoring pairs."""
    m1, m2, md = monitor_pairs(data)
    with torch.no_grad():
        ratio = (torch.norm(flow(m1)[0] - flow(m2)[0], dim=1) / md).numpy()
    return np.percentile(ratio, [10, 50, 90]), ratio


def train_jacobian(C, data, epochs, steps_per_epoch=79, batch=256, lr=5e-4, verbose_every=20,
                   best_path='flow_best.pt'):
    """isobank's flow and optimiser, trained with the local metric loss only."""
    x = torch.as_tensor(data['train'], dtype=torch.float32)
    g = torch.as_tensor(data['g_train'], dtype=torch.float32)
    flow = IsometricFlow(d=C.d_flow, n_layers=C.n_layers, hidden_dim=C.hidden_dim,
                         masks=C.masks, depth=getattr(C, 'DEPTH', 3))
    optimizer = optim.Adam(flow.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    jac = vmap(jacrev(lambda t: flow(t[None])[0][0]))
    best = -np.inf
    for epoch in range(epochs):
        flow.train()
        for _ in range(steps_per_epoch):
            idx = torch.randint(0, len(x), (batch,))
            J = jac(x[idx])                                                  # (B, D, D)
            loss = torch.linalg.matrix_norm(torch.einsum('bai,baj->bij', J, J) - g[idx]).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()
        if epoch % verbose_every == 0:
            flow.eval()
            (p10, p50, p90), _ = score(flow, data)
            print(f"Epoch {epoch:04d} | metric loss {loss.item():.3e} | "
                  f"monitor p10={p10:.4f} med={p50:.4f} p90={p90:.4f}")
            sys.stdout.flush()
            if -abs(p10 - 1) > best:                                         # as train_flow.py
                best = -abs(p10 - 1)
                torch.save(flow.state_dict(), best_path)
    flow.load_state_dict(torch.load(best_path, weights_only=True))
    return flow


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='default: autoencoder_comparison/runs/<case>')
    ap.add_argument('--loss', choices=['full', 'weights', 'jacobian'], required=True)
    ap.add_argument('--weights', type=float, nargs=4, default=None,
                    metavar=('W_GLOBAL', 'W_LOCAL', 'W_BOUND', 'W_EDGE'))
    ap.add_argument('--epochs', type=int, default=None, help="default: the config's N_EPOCHS")
    args = ap.parse_args()

    C = load_config(args.config)
    case = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', case)
    os.makedirs(out, exist_ok=True)
    epochs = args.epochs or C.N_EPOCHS
    tag = args.loss if args.loss != 'weights' else 'weights_' + '_'.join(f'{w:g}' for w in args.weights)
    best_path = os.path.join(out, f'flow_{tag}_best.pt')

    data = make_dataset(C, out)
    t0 = time.time()
    if args.loss == 'jacobian':
        flow = train_jacobian(C, data, epochs, best_path=best_path)
    else:
        pairs = make_pairs(C, data, out)
        weights = getattr(C, 'LOSS_WEIGHTS', (1.0, 5.0, 10.0, 10.0))
        if args.loss == 'weights':
            if args.weights is None:
                raise SystemExit("--loss weights needs --weights W_GLOBAL W_LOCAL W_BOUND W_EDGE")
            weights = tuple(args.weights)
        train_isometric_flow(
            C.make_metric(), float(data['r_target']), C.variable_format,
            pairs['long'], pairs['local'], pairs['bound'], monitor_pairs(data),
            d_flow=C.d_flow, n_layers=C.n_layers, hidden_dim=C.hidden_dim, masks=C.masks,
            depth=getattr(C, 'DEPTH', 3), edge_pairs=pairs['edge'], weights=weights,
            tail_w=getattr(C, 'TAIL_W', 5.0), embed_fn=C.embed_fn, phi_col=C.phi_col,
            n_epochs=epochs, best_path=best_path,
            final_path=os.path.join(out, f'flow_{tag}_final.pt'))
        flow, _ = load_checkpoint(best_path)          # written by this run

    (p10, p50, p90), ratio = score(flow, data)
    print(f"\n{tag}: d_z/d_proper p10 {p10:.3f}  median {p50:.3f}  p90 {p90:.3f}"
          f"   [{(time.time() - t0) / 60:.0f} min]")
    np.save(os.path.join(out, f'ratios_flow_{tag}.npy'), ratio)
    with open(os.path.join(out, f'result_flow_{tag}.json'), 'w') as fh:
        json.dump(dict(loss=args.loss, weights=args.weights, epochs=epochs,
                       p10=p10, p50=p50, p90=p90), fh, indent=1)


if __name__ == '__main__':
    main()
