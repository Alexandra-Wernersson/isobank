"""SVD basis and training loops, with the same settings as our full model:
  optimiser   Adam, lr 1e-3, eps 1e-7
  schedule    ReduceLROnPlateau on the validation loss: factor 0.1, patience 125 epochs, relative threshold 1e-3, floor 1e-8
  early stop  once the learning rate is at the floor and a further `patience` epochs bring no improvement
  loss        L1 (mean absolute error); the autoencoder pretraining adds the latent penalty
  batch       256
The real runs use up to 5000 epochs; the defaults below are smaller so the example notebook runs on a laptop CPU in minutes.
Only numpy and torch are needed.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from architectures import FunnelAutoencoder, ParameterToDecoder


# ------------------------------------------------------------------------------------------------ SVD basis
@dataclass
class SVDBasis:
    """a mean-centred truncated SVD basis, fitted once on the training curves and then frozen"""
    mean: np.ndarray        # (n_times,)
    components: np.ndarray  # (n_components, n_times): the leading right-singular vectors
    singular_values: np.ndarray

    def transform(self, x):
        return (x - self.mean) @ self.components.T

    def inverse_transform(self, coefficients):
        return coefficients @ self.components + self.mean


def fit_svd_basis(x: np.ndarray, n_components: int) -> SVDBasis:
    mean = x.mean(axis=0)
    _, s, vt = np.linalg.svd(x - mean, full_matrices=False)
    return SVDBasis(mean=mean, components=vt[:n_components], singular_values=s[:n_components])


@dataclass
class Scaler:
    """targets / one common scale (as our full model scales its phase coefficients by their overall standard deviation), so the
    relative size of the coefficients, which carries the information, is kept"""
    scale: float

    @classmethod
    def fit(cls, y):
        return cls(float(np.std(y)))

    def __call__(self, y):
        return (y / self.scale).astype(np.float32)

    def inverse(self, y):
        return np.asarray(y, np.float64) * self.scale


# ------------------------------------------------------------------------------------------------ the optimiser and its schedule
@dataclass
class Recipe:
    epochs: int = 600
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 125
    threshold: float = 1e-3
    min_lr: float = 1e-8
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print_every: int = 100


def make_optimizer(params, r: Recipe):
    opt = torch.optim.Adam(params, lr=r.lr, eps=1e-7)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=r.patience, min_lr=r.min_lr, threshold=r.threshold,
                                                       threshold_mode="rel")
    return opt, sched


def plateaued(opt, sched, r: Recipe) -> bool:
    """the learning rate is at its floor and the scheduler has waited a full patience window without improvement"""
    return opt.param_groups[0]["lr"] <= r.min_lr * 1.0001 and sched.num_bad_epochs >= r.patience


def _batches(n, batch_size, generator):
    idx = torch.randperm(n, generator=generator)
    return [idx[i:i + batch_size] for i in range(0, n, batch_size)]


# ------------------------------------------------------------------------------------------------ direct regression: theta -> targets
def train_direct(model: nn.Module, x, y, x_val, y_val, r: Recipe = Recipe(), name="direct", seed=0):
    """L1 regression of y on x; returns the model at its best validation epoch and the loss history"""
    g = torch.Generator().manual_seed(seed)
    dev = r.device
    model = model.to(dev)
    X, Y = torch.as_tensor(x, device=dev), torch.as_tensor(y, device=dev)
    Xv, Yv = torch.as_tensor(x_val, device=dev), torch.as_tensor(y_val, device=dev)
    opt, sched = make_optimizer([p for p in model.parameters() if p.requires_grad], r)
    hist, best, best_state, t0 = {"train": [], "val": []}, float("inf"), None, time.time()
    for epoch in range(1, r.epochs + 1):
        model.train()
        tot = 0.0
        for b in _batches(len(X), r.batch_size, g):
            opt.zero_grad()
            loss = torch.abs(model(X[b]) - Y[b]).mean()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = torch.abs(model(Xv) - Yv).mean().item()
        sched.step(val)
        hist["train"].append(tot / len(X)); hist["val"].append(val)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % r.print_every == 0:
            print(f"[{name}] epoch {epoch:5d} | train {hist['train'][-1]:.3e} | val {val:.3e} | lr {opt.param_groups[0]['lr']:.0e} | {time.time() - t0:.0f} s")
        if plateaued(opt, sched, r):
            print(f"[{name}] early stop at epoch {epoch}: learning rate at the floor, no improvement for {r.patience} epochs")
            break
    model.load_state_dict(best_state)
    print(f"[{name}] best validation loss {best:.3e}")
    return model.cpu().eval(), hist


# ------------------------------------------------------------------------------------------------ autoencoder: two stages
def pretrain_autoencoder(y, y_val, latent=32, hidden=64, regulariser=1e-3, r: Recipe = Recipe(), name="autoencoder", seed=0):
    """stage 1: the autoencoder on the targets alone, L1 reconstruction + latent penalty"""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    dev = r.device
    ae = FunnelAutoencoder(y.shape[1], latent, hidden, regulariser).to(dev)
    Y, Yv = torch.as_tensor(y, device=dev), torch.as_tensor(y_val, device=dev)
    opt, sched = make_optimizer(ae.parameters(), r)
    hist, t0 = {"train": [], "val": []}, time.time()
    for epoch in range(1, r.epochs + 1):
        ae.train()
        tot = 0.0
        for b in _batches(len(Y), r.batch_size, g):
            opt.zero_grad()
            rec, pen = ae(Y[b])
            loss = torch.abs(rec - Y[b]).mean() + pen
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
        ae.eval()
        with torch.no_grad():
            rec, pen = ae(Yv)
            val = torch.abs(rec - Yv).mean().item() + pen.item()
        sched.step(val)
        hist["train"].append(tot / len(Y)); hist["val"].append(val)
        if epoch % r.print_every == 0:
            print(f"[{name} pretrain] epoch {epoch:5d} | val reconstruction + penalty {val:.3e} | lr {opt.param_groups[0]['lr']:.0e} | {time.time() - t0:.0f} s")
        if plateaued(opt, sched, r):
            print(f"[{name} pretrain] early stop at epoch {epoch}")
            break
    return ae.cpu().eval(), hist


def train_two_stage_autoencoder(x, y, x_val, y_val, latent=32, hidden=64, regulariser=1e-3, pretrain: Recipe = Recipe(),
                                second: Recipe = Recipe(), freeze_decoder=False, name="autoencoder", seed=0):
    """stage 1 (pretrain the autoencoder on y), then stage 2: the parameter network theta -> z onto the pretrained decoder, trained by
    L1 regression of y on theta (the decoder keeps training unless freeze_decoder). Returns the theta -> y model and both histories."""
    ae, h1 = pretrain_autoencoder(y, y_val, latent, hidden, regulariser, pretrain, name, seed)
    torch.manual_seed(seed + 1)
    model = ParameterToDecoder(ae.decoder, x.shape[1], latent, hidden, freeze_decoder)
    model, h2 = train_direct(model, x, y, x_val, y_val, second, name=f"{name} parameters", seed=seed)
    return model, {"pretrain": h1, "regression": h2}


def predict(model: nn.Module, x) -> np.ndarray:
    with torch.no_grad():
        return model(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
