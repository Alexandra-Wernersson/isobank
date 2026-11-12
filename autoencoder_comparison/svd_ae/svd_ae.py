"""
svd_ae.py -- the full pipeline of Cabass et al. on the eccentric 5D space (phase-SVD
coefficients -> distance-preserving autoencoder -> lattice in the latent space), scored
exactly as svd_baseline/svd_grid_refined.py scores the plain SVD lattice.

    python autoencoder_comparison/svd_ae/svd_ae.py data     --out DIR
    python autoencoder_comparison/svd_ae/svd_ae.py train    --out DIR --latent-dim 3
    python autoencoder_comparison/svd_ae/svd_ae.py count    --out DIR --parts 50 --part 7
    python autoencoder_comparison/svd_ae/svd_ae.py evaluate --out DIR --isobank-ff FF.npy --isobank-bank BANK.npy

data      waveforms, SVD basis, coefficients and the parameters -> coefficients regressor, as
          svd_grid_refined.py (same draws, seeds and regressor); 10 coefficients are kept (Cabass: dims = 10).
train     the autoencoder of Cabass's GW example (autoencoder_GWs, autoencoder_use_example/
          train_autoencoders_b_1_sb_0_to_2.py): MLP 10 -> latent -> 10 (GELU, 64 hidden, 4 layers),
          coefficients divided by std(c0), half the waveforms held out (seed 1648), loss = reconstruction
          MSE + MSE of the exp(-d)-weighted pairwise distances, Adam 1e-3, 50,000 steps, best validation
          model. Two changes: the pairs are all pairs of a random batch of 2048 waveforms per step (all
          pairs of 16,000 waveforms do not fit in memory), and float64, so precision is not a confound.
count     bank size = lattice cells reached, sampled as svd_grid_refined.py (regressor; 80% of the draws
          uniform in Mc^(-5/3)), in independent parts: the plain SVD lattice (BCC on c0..c2, as
          svd_grid_refined.py) and the latent lattices (BCC in 3D, hexagonal in 2D).
evaluate  fitting factor of each of the 4,000 injections (first 4,000 of seed 4002, as the isobank
          fitting factors) with its lattice template, as svd_grid_refined.py: overlap with the signal's
          own amplitude and one template amplitude profile, no extra time/phase maximisation. Table of
          all lattices, and the cumulative plot at the bank size closest to isobank's.

Spacings are in coefficient units (1 - M ~ |dc|^2 / 2) for every lattice: BCC cubic constant a,
hexagonal nearest-neighbour distance a; the latent lattices use a / std(c0), the units of the latent space.
"""
import argparse, json, os, subprocess, sys, time
from multiprocessing import Pool
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, 'svd_baseline'))
sys.path.insert(0, os.path.dirname(HERE))
_argv, sys.argv = sys.argv, [sys.argv[0], 'ecc5d']     # svd_feasibility.py reads the space from argv[1]
import svd_feasibility as F
import svd_grid_estimate as G
import svd_grid_refined as RF
sys.argv = _argv
from train_autoencoder import Autoencoder                 # the MLP autoencoder of Cabass et al.

N_COEF = 10                                    # Cabass: dims = 10
SVD_BCC = RF.BCC                               # 0.44 ... 0.60, as svd_grid_refined.py
LATENT = {3: [.40, .44, .48, .52, .56, .60],   # BCC
          2: [.06, .08, .10, .13, .16, .20, .25, .30]}   # hexagonal
N_SAMPLES, CHUNK = 200_000_000, 500_000        # 2e8 draws, as svd_grid_refined.py
BITS = 20


def ncpu():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', 8))


# --- lattices --------------------------------------------------------------------------------

def step_vec(dim, a):
    return np.full(3, a) if dim == 3 else np.array([a, a * np.sqrt(3)])


def snap(z, step):
    """Nearest point of step * Z^d  U  step * (Z^d + 1/2): body-centred cubic for step = (a, a, a)
    (identical to svd_grid_refined.snap_bcc), hexagonal of side a for step = (a, a sqrt 3).
    Returns the points and one int64 id per cell."""
    i0 = np.rint(z / step); i1 = np.rint(z / step - .5)
    p0, p1 = i0 * step, (i1 + .5) * step
    odd = ((z - p1) ** 2).sum(1) < ((z - p0) ** 2).sum(1)
    p = np.where(odd[:, None], p1, p0)
    u = np.where(odd[:, None], i1, i0).astype(np.int64) + (1 << (BITS - 1))
    if u.min() < 0 or u.max() >= 1 << BITS:
        raise ValueError('lattice index out of range')
    ids = odd.astype(np.int64)
    for j in range(u.shape[1]):
        ids |= u[:, j] << (1 + BITS * j)
    return p, ids


def lattices(latent_dims):
    """name -> (kind, latent dim or None, spacing in coefficient units)."""
    out = {f'svd_a{a:.2f}': ('svd', None, a) for a in SVD_BCC}
    for k in latent_dims:
        out.update({f'ae{k}_a{a:.2f}': ('ae', k, a) for a in LATENT[k]})
    return out


# --- models ----------------------------------------------------------------------------------

def load_regressor(out):
    r = torch.load(os.path.join(out, 'regressor.pt'), weights_only=True)
    net = nn.Sequential(*[m for i in range(4) for m in (nn.Linear(8 if i == 0 else 256, 256), nn.GELU())],
                        nn.Linear(256, RF.KMAX))
    net.load_state_dict(r['state']); net = net.float().eval()
    mx, sx, my, sy = (r[k] for k in ('mx', 'sx', 'my', 'sy'))

    def predict(p):    # as svd_grid_refined.py
        with torch.no_grad():
            return net((RF.feats(p, torch.float32) - mx.float()) / sx.float()).double().numpy() * sy.numpy() + my.numpy()
    return predict


def load_ae(out, k):
    d = torch.load(os.path.join(out, f'svd_ae_latent{k}.pt'), weights_only=True)
    model = Autoencoder(N_COEF, k).double(); model.load_state_dict(d['state']); model.eval()
    norm = float(d['norm'])

    def enc(c):
        with torch.no_grad():
            return model.encoder(torch.tensor(c / norm)).numpy()

    def dec(z):
        with torch.no_grad():
            return model.decoder(torch.tensor(z)).numpy() * norm
    return enc, dec, norm


# --- steps -----------------------------------------------------------------------------------

def data(args):
    """svd_grid_refined.py up to the regressor; keeps 10 coefficients and saves everything."""
    n_cpu = ncpu(); torch.set_num_threads(n_cpu); torch.manual_seed(0)
    f_psd, s = np.loadtxt(F.S['psd']).T[:2]
    psd = np.interp(F.GRID, f_psd, s ** 2 if F.S['asd'] else s)
    tr, box, inj = F.draw(G.N_TRAIN, 21), G.box_points(), G.injections()
    extra = RF.sample_u(RF.N_EXTRA, np.random.default_rng(77))
    n_ep = 20000                                  # as svd_grid_refined.py
    if args.smoke:      # a quick local check of the whole chain, not for results
        tr, extra, inj, n_ep = tr[:300], extra[:300], inj[:200], 200
    X = np.vstack([tr, box, extra, inj]); n1 = len(tr); nfit = n1 + len(box) + len(extra)
    t0 = time.time()
    with Pool(n_cpu) as pool:
        out = pool.map(F.waveform, X, chunksize=16)
    A = np.array([o[0] for o in out]); P = np.array([o[1] for o in out]); del out
    print(f'{len(X):,} waveforms: {(time.time() - t0) / 60:.1f} min', flush=True)
    W = A / np.sqrt(psd); W /= np.sqrt((W ** 2).sum(1, keepdims=True)); del A
    w = W[:n1].mean(0); w /= np.sqrt((w ** 2).sum())
    Q, _ = np.linalg.qr(np.c_[np.ones_like(F.GRID), F.GRID - F.GRID.mean()] * w[:, None])
    mean = P[:n1].mean(0)
    for _ in range(3):
        R = (P - mean) * w; R -= (R @ Q) @ Q.T
        mean = mean + R[:n1].mean(0) / np.where(w > 1e-12, w, np.inf)
    R = (P - mean) * w; R -= (R @ Q) @ Q.T; del P
    _, sv, Vt = np.linalg.svd(R[:n1], full_matrices=False)
    C = R @ Vt[:N_COEF].T
    print('top sv', np.round(sv[:6], 3), '  coefficient std', np.round(C[:nfit].std(0), 4), flush=True)

    # regressor parameters -> top 5 coefficients: svd_grid_refined.py, unchanged
    Xt = RF.feats(X[:nfit], torch.float64); Ct = torch.tensor(C[:nfit, :RF.KMAX], dtype=torch.float64)
    mx, sx, my, sy = Xt.mean(0), Xt.std(0), Ct.mean(0), Ct.std(0); sx[sx == 0] = 1
    net = nn.Sequential(nn.Linear(8, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU(),
                        nn.Linear(256, 256), nn.GELU(), nn.Linear(256, RF.KMAX)).double()
    opt = torch.optim.Adam(net.parameters(), 2e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_ep)
    xn, yn = (Xt - mx) / sx, (Ct - my) / sy
    for ep in range(n_ep):
        opt.zero_grad(); l = ((net(xn) - yn) ** 2).mean(); l.backward(); opt.step(); sch.step()
        if ep % 4000 == 0: print(ep, float(l), time.strftime('%X'), flush=True)
    os.makedirs(args.out, exist_ok=True)
    torch.save(dict(state=net.state_dict(), mx=mx, sx=sx, my=my, sy=sy), os.path.join(args.out, 'regressor.pt'))
    err = load_regressor(args.out)(inj) - C[nfit:, :RF.KMAX]
    rms = np.sqrt((err ** 2).mean(0))
    print('regressor rms on injections', np.round(rms, 4), '(svd_refined_ecc5d.json: 0.2944 0.0029 0.0008 0.001 0.0001)', flush=True)
    mc = (inj[:, 0] * inj[:, 1]) ** .6 / (inj[:, 0] + inj[:, 1]) ** .2
    np.savez(os.path.join(args.out, 'svd_data.npz'), Vt=Vt[:N_COEF], w=w, C=C, X=X, n1=n1, nfit=nfit,
             r_inj=R[nfit:], W_inj=W[nfit:], mc_inj=mc, sv=sv[:12], regressor_rms=rms)
    print(f"-> {os.path.join(args.out, 'svd_data.npz')}")


def train(args):
    """Cabass's GW-example autoencoder on the coefficients of the training waveforms."""
    torch.set_num_threads(ncpu()); torch.manual_seed(args.latent_dim)
    Z = np.load(os.path.join(args.out, 'svd_data.npz'))
    c = Z['C'][:int(Z['nfit'])]                 # training waveforms only, never the injections
    norm = c[:, 0].std()
    x = torch.tensor(c / norm)
    np.random.seed(1648)
    held = np.random.choice([0, 1], size=(len(x),), p=[1. / 2, 1. / 2]).astype('bool')
    xt, xv = x[~held], x[held]
    xv = xv[torch.randperm(len(xv))[:args.batch]]
    model = Autoencoder(N_COEF, args.latent_dim).double()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = torch.nn.MSELoss()

    def loss(b):
        rec, z = model(b)
        d0 = torch.pdist(b); wt = torch.exp(-d0)
        return mse(rec, b) + mse(wt * torch.pdist(z), wt * d0)

    print(f'autoencoder {N_COEF} -> {args.latent_dim} -> {N_COEF}, {len(xt):,} training and {len(x) - len(xt):,} '
          f'held-out waveforms, std(c0) = {norm:.2f}', flush=True)
    best, history, t0 = (np.inf, None), dict(step=[], train=[], valid=[]), time.time()
    for step in range(args.steps):
        opt.zero_grad()
        l = loss(xt[torch.randint(len(xt), (args.batch,))]); l.backward(); opt.step()
        if step % 100 == 0 or step == args.steps - 1:
            with torch.no_grad():
                lv = float(loss(xv))
            history['step'].append(step); history['train'].append(float(l)); history['valid'].append(lv)
            if lv < best[0]:
                best = (lv, {k: v.clone() for k, v in model.state_dict().items()})
            if step % 5000 == 0:
                print(f'step {step}  train {float(l):.3e}  valid {lv:.3e}  [{(time.time() - t0) / 60:.0f} min]', flush=True)
    torch.save(dict(state=best[1], norm=torch.tensor(norm)), os.path.join(args.out, f'svd_ae_latent{args.latent_dim}.pt'))
    with open(os.path.join(args.out, f'history_svd_ae_latent{args.latent_dim}.json'), 'w') as fh:
        json.dump(history, fh)
    print(f'best validation loss {best[0]:.3e} -> svd_ae_latent{args.latent_dim}.pt')


def count(args):
    """Distinct lattice cells reached by this part's share of the 2e8 draws."""
    torch.set_num_threads(1)
    predict = load_regressor(args.out)
    aes = {k: load_ae(args.out, k) for k in args.latent_dims}
    lat = lattices(args.latent_dims)
    n = args.n_samples // args.parts
    rng = np.random.default_rng([6, args.part])
    cells = {name: np.empty(0, np.int64) for name in lat}
    t0 = time.time()
    for done in range(0, n, CHUNK):
        c5 = predict(RF.sample(min(CHUNK, n - done), rng))
        c = np.c_[c5, np.zeros((len(c5), N_COEF - RF.KMAX))]    # the regressor gives 5; beyond the 5th, < 1e-6 of mismatch (svd_feasibility k = 5)
        z = {k: aes[k][0](c) for k in aes}
        for name, (kind, k, a) in lat.items():
            ids = snap(c[:, :3], step_vec(3, a))[1] if kind == 'svd' else snap(z[k], step_vec(k, a) / aes[k][2])[1]
            cells[name] = np.union1d(cells[name], ids)
        print(f'{done + len(c5):,} of {n:,}  [{(time.time() - t0) / 60:.1f} min]', flush=True)
    os.makedirs(os.path.join(args.out, 'cells'), exist_ok=True)
    np.savez(os.path.join(args.out, 'cells', f'part_{args.part:04d}.npz'), n_samples=n, **cells)


def evaluate(args):
    Z = np.load(os.path.join(args.out, 'svd_data.npz'))
    Vt, w, nfit = Z['Vt'], Z['w'], int(Z['nfit'])
    ci, ri, Wi, mc = Z['C'][nfit:], Z['r_inj'], Z['W_inj'], Z['mc_inj']
    mask = w > 1e-12
    aes = {k: load_ae(args.out, k) for k in args.latent_dims}
    lat = lattices(args.latent_dims)

    def ff_of(coef):       # svd_grid_refined.coverage: signal amplitude x template amplitude x cos(phase error)
        rec = coef @ Vt[:coef.shape[1]]
        d = np.zeros_like(rec); d[:, mask] = (rec - ri)[:, mask] / w[mask]
        return (Wi * w * np.cos(d)).sum(1)

    # bank sizes: union of the parts' cells, after half of the parts and after all
    files = [os.path.join(args.out, 'cells', f'part_{i:04d}.npz') for i in range(args.parts)]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise SystemExit(f'{len(missing)} parts missing, e.g. {missing[0]}')
    cells, half, n_drawn = {name: np.empty(0, np.int64) for name in lat}, {}, 0
    for i, f in enumerate(files):
        with np.load(f) as p:
            n_drawn += int(p['n_samples'])
            for name in lat:
                cells[name] = np.union1d(cells[name], p[name])
        if i == args.parts // 2 - 1:
            half = {name: len(v) for name, v in cells.items()}
    print(f'bank sizes from {n_drawn:,} draws (half: {n_drawn // 2:,})')

    rows, ffs = [], {}
    for name, (kind, k, a) in lat.items():
        if kind == 'svd':
            coef = snap(ci[:, :3], step_vec(3, a))[0]
        else:
            enc, dec, norm = aes[k]
            coef = dec(snap(enc(ci), step_vec(k, a) / norm)[0])
        ffs[name] = ff_of(coef)
        rows.append(dict(name=name, kind=kind, k=k, a=a, n=len(cells[name]), n_half=half[name]))
    for k, (enc, dec, norm) in aes.items():      # autoencoder alone, no lattice: its reconstruction error
        ffs[f'ae{k}_nolattice'] = ff_of(dec(enc(ci)))
        rows.append(dict(name=f'ae{k}_nolattice', kind='ae', k=k, a=0., n=0, n_half=0))
    for r in rows:
        ff = ffs[r['name']]; np.save(os.path.join(args.out, f"ff_{r['name']}.npy"), ff)
        r.update(cov=100 * (ff >= .97).mean(), cov_mc14=100 * (ff[mc < 1.4] >= .97).mean(),
                 p1=np.percentile(ff, 1), worst=ff.min())

    ref = json.load(open(os.path.join(ROOT, 'svd_baseline', 'results', 'svd_refined_ecc5d.json')))
    ref = {r['spacing']: r for r in ref['rows'] if r['scheme'] == 'bcc'}
    n_iso = len(np.load(args.isobank_bank, mmap_mode='r'))
    lines = [f'Lattice banks, first 4,000 injections of seed 4002, FF >= 0.97. Bank size from {n_drawn:,} draws '
             f'(and from half of them, for convergence). isobank: {n_iso:,} templates.', '',
             '| Lattice | Spacing | Templates (half the draws) | Coverage | Mc < 1.4 | 1st pct | Worst | svd_refined_ecc5d.json |',
             '|---|---|---|---|---|---|---|---|']
    names = {'svd': 'SVD, BCC on c0..c2', 3: 'SVD + autoencoder, latent 3, BCC', 2: 'SVD + autoencoder, latent 2, hexagonal'}
    for r in rows:
        label = names['svd'] if r['kind'] == 'svd' else names[r['k']]
        if r['a'] == 0:
            label, size, sp = label.split(',')[0] + f", latent {r['k']}, no lattice", '–', '–'
        else:
            size, sp = f"{r['n']:,} ({r['n_half']:,})", f"{r['a']:.2f}"
        old = ref.get(r['a']) if r['kind'] == 'svd' else None
        old = f"{old['occ']['200000000']:,}, {100 * old['cov']:.2f}%" if old else ''
        lines.append(f"| {label} | {sp} | {size} | {r['cov']:.2f}% | {r['cov_mc14']:.1f}% | {r['p1']:.4f} "
                     f"| {r['worst']:.4f} | {old} |")
    table = '\n'.join(lines) + '\n'
    with open(os.path.join(args.out, 'lattice_table.md'), 'w') as fh:
        fh.write(table)
    print(table)

    # the cumulative plot: each lattice at the bank size closest to isobank's
    def human(n):
        return f'{n / 1e6:.2f}M' if n >= 1e6 else f'{n / 1e3:.0f}k'

    def first_4000(path, tag):
        ff = np.load(path)
        if len(ff) < 4000:
            raise SystemExit(f'{path}: {len(ff)} fitting factors, need the first 4,000 of seed 4002')
        out = os.path.join(args.out, f'ff_{tag}_first4000.npy'); np.save(out, ff[:4000]); return out

    specs = [f"{first_4000(args.isobank_ff, 'isobank')}:isobank ({human(n_iso)})"]
    for kind, k, label in (('svd', None, 'SVD lattice, no autoencoder'), *[('ae', k, f'SVD + autoencoder, latent {k}')
                                                                            for k in sorted(aes, reverse=True)]):
        cand = [r for r in rows if r['kind'] == kind and r['k'] == k and r['n'] > 0]
        r = min(cand, key=lambda r: abs(np.log(r['n'] / n_iso)))
        specs.append(f"{os.path.join(args.out, 'ff_' + r['name'] + '.npy')}:{label} ({human(r['n'])})")
    for i, spec in enumerate(args.extra_ff):
        path, _, label = spec.partition(':')
        specs.append(f"{first_4000(path, f'extra{i}')}:{label}")
    subprocess.run([sys.executable, os.path.join(ROOT, 'figures', 'plot_ff_cdf.py'),
                    '--out', os.path.join(args.out, 'ff_cdf_svd.pdf'), '--xlim', args.xlim,
                    *[a for s in specs for a in ('--ff', s)]], check=True)
    summary = subprocess.run([sys.executable, os.path.join(ROOT, 'autoencoder_comparison', 'summarise_ff.py'), *specs],
                             check=True, capture_output=True, text=True).stdout
    with open(os.path.join(args.out, 'ff_summary_svd.md'), 'w') as fh:
        fh.write(summary)
    print(summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='mode', required=True)
    p = {m: sub.add_parser(m) for m in ('data', 'train', 'count', 'evaluate')}
    for q in p.values():
        q.add_argument('--out', required=True, help='folder of this pipeline\'s files')
    p['data'].add_argument('--smoke', action='store_true', help='a few hundred waveforms: checks the chain only')
    p['train'].add_argument('--latent-dim', type=int, required=True, choices=sorted(LATENT))
    p['train'].add_argument('--steps', type=int, default=50_000)
    p['train'].add_argument('--batch', type=int, default=2048)
    for m in ('count', 'evaluate'):
        p[m].add_argument('--latent-dims', type=int, nargs='+', default=[3, 2])
        p[m].add_argument('--parts', type=int, required=True)
    p['count'].add_argument('--part', type=int, required=True)
    p['count'].add_argument('--n-samples', type=int, default=N_SAMPLES, help='over all parts')
    p['evaluate'].add_argument('--isobank-ff', required=True, help='evaluate_bank.py --save-ff, seed 4002')
    p['evaluate'].add_argument('--isobank-bank', required=True)
    p['evaluate'].add_argument('--extra-ff', action='append', default=[],
                               help='path:label, fitting factors on the seed-4002 injections (first 4,000 used)')
    p['evaluate'].add_argument('--xlim', default='0.85,1.0')
    args = ap.parse_args()
    dict(data=data, train=train, count=count, evaluate=evaluate)[args.mode](args)


if __name__ == '__main__':
    main()
