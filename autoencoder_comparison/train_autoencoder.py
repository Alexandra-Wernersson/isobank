"""
train_autoencoder.py -- the distance-preserving autoencoder of Cabass et al.
(github.com/giovanni-cabass/autoencoder_GWs, Appendix B) on an isobank space.

    python autoencoder_comparison/train_autoencoder.py --config configs/eccentric_5d.py \
        --latent-dim 5 --side decoder --epochs 2000

Same data, metric and held-out monitoring pairs as scripts/train_flow.py, so the
score is the one isobank reports for its flow: percentiles of d_z / d_proper on
monitoring pairs at placement scale (validated isobank flow: p10 = 0.96).

The model and loss are those of the Appendix B example: MLP encoder and decoder
(GELU, 64 hidden, 4 layers), AdamW, loss = reconstruction MSE + mean Frobenius
norm of (induced metric - target). The coordinates are scaled to [-1, 1] and
the metric with them. Two ways to impose the metric:

  * --side decoder (Appendix B): the metric the decoder induces on the latent
    space, J_D^T M J_D, must be the identity (flat latent space);
  * --side encoder: the metric the encoder pulls back, J_E^T J_E, must be M.

With --latent-dim below the dimension of the space (as in the GW example,
10 -> 2) only the decoder side applies: directions shorter than the template
spacing can be dropped.

The data, metrics, r_target and pairs are computed once and cached in --out.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from torch.func import jacrev, vmap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from isobank import calibrate_r_target, metric_guided_pairs, filter_by_distance


def load_config(path):
    spec = importlib.util.spec_from_file_location('config', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_mlp(input_dim, hidden_dim, output_dim, layers, activation=nn.GELU()):
    """As in the Appendix B example."""
    seq = [nn.Linear(input_dim, hidden_dim), activation]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, layers=4):
        super().__init__()
        self.encoder = build_mlp(input_dim, hidden_dim, latent_dim, layers)
        self.decoder = build_mlp(latent_dim, hidden_dim, input_dim, layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def make_dataset(C, out):
    """Training points and metrics, r_target and monitoring pairs, as train_flow.py."""
    cache = os.path.join(out, 'dataset.npz')
    if os.path.exists(cache):
        print(f"using cached data: {cache}")
        with np.load(cache, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}

    metric = C.make_metric()
    valid_fn = None if C.valid_mask is None else (lambda t: bool(C.valid_mask(t[None])[0]))

    print("calibrating r_target..."); sys.stdout.flush()
    r_target, _ = calibrate_r_target(metric, C.lo, C.hi, mu=C.MU, n_ref=C.N_REF,
                                     valid_fn=valid_fn)
    print(f"r_target = {r_target:.5f}")

    train, val = C.sample_data()
    print(f"metrics at {len(train)} + {len(val)} points..."); sys.stdout.flush()
    g_train = np.asarray(metric.get_metric(torch.tensor(train), metric_type='hessian'))
    g_val = np.asarray(metric.get_metric(torch.tensor(val), metric_type='hessian'))

    print("monitoring pairs..."); sys.stdout.flush()
    mt1, mt2 = metric_guided_pairs(val, g_val, r_target, C.N_MON, C.BOUND_FRACS,
                                   C.lo, C.hi, seed=999, phi_col=C.phi_col,
                                   valid_fn=valid_fn)
    mt1, mt2, md = filter_by_distance(metric, mt1, mt2, 0.5 * r_target, 1.5 * r_target)
    print(f"  {len(md)} pairs")

    data = dict(train=train, val=val, g_train=g_train.astype(np.float64),
                g_val=g_val.astype(np.float64), r_target=np.array(r_target),
                mon1=np.asarray(mt1), mon2=np.asarray(mt2), mon_d=np.asarray(md))
    np.savez(cache, **data)
    return data


class Scaling:
    """theta = mid + half * x, with x in [-1, 1]; the metric scales with it."""

    def __init__(self, lo, hi):
        self.mid = torch.tensor((hi + lo) / 2, dtype=torch.float32)
        self.half = torch.tensor((hi - lo) / 2, dtype=torch.float32)

    def x(self, theta):
        return (torch.as_tensor(theta, dtype=torch.float32) - self.mid) / self.half

    def metric(self, g):
        h = self.half.double()
        return torch.as_tensor(g, dtype=torch.float64) * h[:, None] * h[None, :]


def metric_loss(model, x, g, side):
    """Mean Frobenius norm of (induced metric - target) over the batch."""
    if side == 'decoder':
        z = model.encoder(x)
        J = vmap(jacrev(model.decoder))(z)                          # (B, D, k)
        induced = torch.einsum('bia,bij,bjc->bac', J, g, J)
        target = torch.eye(J.shape[-1]).expand_as(induced)
    else:
        J = vmap(jacrev(model.encoder))(x)                          # (B, k, D)
        induced = torch.einsum('bai,baj->bij', J, J)
        target = g
    return torch.linalg.matrix_norm(induced - target).mean()


def monitor(model, scaling, data):
    """Percentiles (10, 50, 90) of d_z / d_proper on the monitoring pairs."""
    with torch.no_grad():
        z1 = model.encoder(scaling.x(data['mon1']))
        z2 = model.encoder(scaling.x(data['mon2']))
    ratio = torch.norm(z1 - z2, dim=1).numpy() / data['mon_d']
    return np.percentile(ratio, [10, 50, 90]), ratio


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', default=None, help='default: autoencoder_comparison/runs/<case>')
    ap.add_argument('--latent-dim', type=int, default=None, help='default: the dimension of the space')
    ap.add_argument('--side', choices=['decoder', 'encoder'], default='decoder')
    ap.add_argument('--epochs', type=int, default=2000)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--alpha', type=float, default=1.0, help='weight of the reconstruction loss')
    ap.add_argument('--beta', type=float, default=1.0, help='weight of the metric loss')
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--monitor-every', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prepare-only', action='store_true', help='only build the cached data')
    args = ap.parse_args()

    C = load_config(args.config)
    case = os.path.splitext(os.path.basename(args.config))[0]
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs', case)
    os.makedirs(out, exist_ok=True)
    k = args.latent_dim or C.D
    if k < C.D and args.side == 'encoder':
        raise SystemExit("--side encoder needs --latent-dim equal to the dimension of the space")

    data = make_dataset(C, out)
    if args.prepare_only:
        return
    scaling = Scaling(C.lo, C.hi)
    x_train = scaling.x(data['train'])
    g_train = scaling.metric(data['g_train']).float()
    ev = np.linalg.eigvalsh(g_train.double().numpy())
    print(f"{C.CASE_NAME}: {len(x_train)} training points, {len(data['mon_d'])} monitoring pairs")
    print(f"scaled metric eigenvalues: min {ev.min():.1e}  max {ev.max():.1e}  "
          f"median condition {np.median(ev[:, -1] / np.abs(ev[:, 0])):.1e}")

    torch.manual_seed(args.seed)
    model = Autoencoder(C.D, k, args.hidden, args.layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    mse = nn.MSELoss()
    tag = f"latent{k}_{args.side}"
    print(f"autoencoder {C.D} -> {k} -> {C.D}, metric imposed on the {args.side}\n")

    history, best = dict(epoch=[], loss=[], recon=[], metric=[], p10=[], p50=[], p90=[]), None
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(x_train))
        totals = np.zeros(3)
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            x, g = x_train[idx], g_train[idx]
            optimizer.zero_grad()
            recon = mse(model(x)[0], x)
            mloss = metric_loss(model, x, g, args.side)
            loss = args.alpha * recon + args.beta * mloss
            loss.backward()
            optimizer.step()
            totals += np.array([loss.item(), recon.item(), mloss.item()]) * len(idx)
        totals /= len(perm)

        if epoch % args.monitor_every == 0 or epoch == args.epochs:
            (p10, p50, p90), _ = monitor(model, scaling, data)
            for key, v in zip(history, (epoch, *totals, p10, p50, p90)):
                history[key].append(float(v))
            if best is None or abs(p10 - 1) < abs(best[1] - 1):    # as train_flow.py
                best = (epoch, p10)
                torch.save(model.state_dict(), os.path.join(out, f'autoencoder_{tag}.pt'))
            print(f"epoch {epoch:5d}  loss {totals[0]:.3e} (recon {totals[1]:.2e}, "
                  f"metric {totals[2]:.2e})  d_z/d_proper p10 {p10:.3f} median {p50:.3f} "
                  f"p90 {p90:.3f}  [{(time.time() - t0) / 60:.1f} min]")
            sys.stdout.flush()

    model.load_state_dict(torch.load(os.path.join(out, f'autoencoder_{tag}.pt'), weights_only=True))
    (p10, p50, p90), ratio = monitor(model, scaling, data)
    print(f"\nbest (epoch {best[0]}): d_z/d_proper p10 {p10:.3f}  median {p50:.3f}  p90 {p90:.3f}"
          f"   (isobank flow: p10 0.96)")
    np.save(os.path.join(out, f'ratios_{tag}.npy'), ratio)
    with open(os.path.join(out, f'history_{tag}.json'), 'w') as fh:
        json.dump(dict(args=vars(args), best_epoch=best[0], p10=p10, p50=p50, p90=p90,
                       history=history), fh, indent=1)


if __name__ == '__main__':
    main()
