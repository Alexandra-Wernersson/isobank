"""
isobank.flow
============

The isometric normalizing flow: a RealNVP-style stack of affine coupling
layers trained so that Euclidean distance in the latent space z matches the
proper distance of the waveform-match metric:

	||flow(theta1) - flow(theta2)||  ~  d_proper(theta1, theta2)

Also provides :func:`wrap_phi`, which wraps a periodic coordinate (e.g. a
reference phase) to the shortest path before any proper-distance computation
(mbank's proper_distance does NOT wrap).

"""

import numpy as np
import torch
import torch.nn as nn

__all__ = ['AffineCouplingLayer', 'IsometricFlow', 'default_masks',
           'wrap_phi', 'wrap_phi_np',
           'save_checkpoint', 'load_checkpoint']

####################################################################################################################
#	Coordinate helpers
####################################################################################################################

def wrap_phi(t1: torch.Tensor, t2: torch.Tensor, phi_col: int) -> torch.Tensor:
	"""
	Return a copy of ``t2`` with the phi column wrapped so that
	|phi2 - phi1| <= pi. Must be applied before every proper_distance call on
	spaces with a periodic reference phase: the metric proper distance is
	computed from the raw coordinate difference and knows nothing about
	periodicity.

	Parameters
	----------
		t1, t2: :class:`torch.Tensor`
			shape (N, D) - the two endpoints of each pair
		phi_col: int
			Column index of the periodic phase coordinate
	"""
	t2w = t2.clone()
	dphi = t2w[:, phi_col] - t1[:, phi_col]
	t2w[:, phi_col] = t1[:, phi_col] + ((dphi + np.pi) % (2 * np.pi) - np.pi)
	return t2w

def wrap_phi_np(phi: np.ndarray) -> np.ndarray:
	"""Wrap phi into [-pi, pi]."""
	return (phi + np.pi) % (2 * np.pi) - np.pi

####################################################################################################################
#	Flow architecture
####################################################################################################################

class AffineCouplingLayer(nn.Module):
	"""
	RealNVP affine coupling layer. The columns selected by ``mask`` condition
	the affine transformation of the remaining columns. The scale is passed
	through tanh for stability and the last linear layer is zero-initialised
	so that the layer starts as the identity.

	``depth`` is the number of hidden (Linear + Tanh) blocks in the
	conditioner: depth=1 gives Linear-Tanh-Linear, depth=3 gives the wider
	3-hidden-layer conditioner used for the HM spaces.
	"""
	def __init__(self, mask, d, hidden_dim=256, depth=3):
		super().__init__()
		self.register_buffer('mask', torch.tensor(mask, dtype=torch.float32))
		d_cond  = int(sum(mask))
		d_trans = d - d_cond
		layers = [nn.Linear(d_cond, hidden_dim), nn.Tanh()]
		for _ in range(depth - 1):
			layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
		layers += [nn.Linear(hidden_dim, 2 * d_trans)]
		self.net = nn.Sequential(*layers)
		nn.init.zeros_(self.net[-1].weight)
		nn.init.zeros_(self.net[-1].bias)

	def _st(self, x):
		cond = x[:, self.mask.bool()]; st = self.net(cond); h = st.shape[1] // 2
		return torch.tanh(st[:, :h]), st[:, h:]

	def forward(self, x):
		s, t = self._st(x); tr = ~self.mask.bool()
		y = x.clone(); y[:, tr] = x[:, tr] * torch.exp(s) + t
		return y, s.sum(-1)

	def inverse(self, y):
		s, t = self._st(y); tr = ~self.mask.bool()
		x = y.clone(); x[:, tr] = (y[:, tr] - t) * torch.exp(-s)
		return x, -s.sum(-1)

def default_masks(d):
	"""
	Default coupling-mask cycle for dimension ``d``. Every dimension is
	transformed at least once per cycle and conditioned on in at least two
	patterns, so all cross-couplings can be learned.
	"""
	if d == 2:
		return [[1, 0], [0, 1]]
	if d == 3:
		return [[1, 1, 0], [0, 1, 1], [1, 0, 1]]
	if d == 4:
		return [[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0]]
	if d == 5:
		return [[1, 1, 0, 0, 0], [0, 0, 1, 1, 1],
		        [1, 0, 1, 0, 1], [0, 1, 0, 1, 0]]
	#generic: alternate halves, then odd/even
	half = d // 2
	m1 = [1] * half + [0] * (d - half)
	m2 = [0] * half + [1] * (d - half)
	m3 = [i % 2 for i in range(d)]
	m4 = [(i + 1) % 2 for i in range(d)]
	return [m1, m2, m3, m4]

class IsometricFlow(nn.Module):
	"""
	Stack of :class:`AffineCouplingLayer` cycling over a list of masks.

	Parameters
	----------
		d: int
			Input dimension (the flow-input dimension, after ``embed_fn``)
		n_layers: int
			Number of coupling layers
		hidden_dim: int
			Width of the conditioner networks
		masks: list
			List of 0/1 masks of length ``d``; layer ``i`` uses
			``masks[i % len(masks)]``. Defaults to :func:`default_masks`.
	"""
	def __init__(self, d, n_layers, hidden_dim=256, masks=None, depth=3):
		super().__init__()
		self.d, self.n_layers, self.hidden_dim, self.depth = d, n_layers, hidden_dim, depth
		self.masks = masks if masks is not None else default_masks(d)
		self.layers = nn.ModuleList([
			AffineCouplingLayer(self.masks[i % len(self.masks)], d, hidden_dim, depth)
			for i in range(n_layers)])

	def forward(self, x):
		ld = torch.zeros(x.shape[0], device=x.device)
		for layer in self.layers: x, l = layer(x); ld += l
		return x, ld

	def inverse(self, z):
		ld = torch.zeros(z.shape[0], device=z.device)
		for layer in reversed(self.layers): z, l = layer.inverse(z); ld += l
		return z, ld

####################################################################################################################
#	Checkpointing
####################################################################################################################

def save_checkpoint(path, flow, r_target, variable_format, **extra):
	"""
	Save a flow checkpoint with all hyperparameters needed to rebuild it via
	:func:`load_checkpoint`, plus any extra metadata (epoch, monitor scores,
	embedding flag, ...).
	"""
	ckpt = {"model_state_dict": flow.state_dict(),
	        "d": flow.d, "n_layers": flow.n_layers,
	        "hidden_dim": flow.hidden_dim, "masks": flow.masks,
	        "depth": flow.depth,
	        "r_target": r_target, "variable_format": variable_format}
	ckpt.update(extra)
	torch.save(ckpt, path)

def load_checkpoint(path, map_location='cpu'):
	"""
	Rebuild an :class:`IsometricFlow` from a checkpoint saved with
	:func:`save_checkpoint`.

	Returns
	-------
		flow: :class:`IsometricFlow`
			The flow in eval() mode
		ckpt: dict
			The full checkpoint dictionary (r_target, variable_format, extras)
	"""
	ckpt = torch.load(path, map_location=map_location, weights_only=False)
	flow = IsometricFlow(d=ckpt['d'], n_layers=ckpt['n_layers'],
	                     hidden_dim=ckpt['hidden_dim'],
	                     masks=ckpt.get('masks'),
	                     depth=ckpt.get('depth', 3))
	flow.load_state_dict(ckpt['model_state_dict'])
	flow.eval()
	return flow, ckpt
