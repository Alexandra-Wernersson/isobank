"""
Generate (once) the training pairs and a fresh held-out test set.

Training pairs reproduce scripts/train_flow.py exactly (same seeds, filters
and counts). Test pairs use new anchors and new seeds and cover every scale:
  global   uniform random pairs over the space
  local    alpha ~ U[0.1, 2.5], no distance filter (alpha stored per pair)
  bound    alpha ~ U[0.6, 1.4], filtered to [0.5, 1.5] r_target
  edge     as bound, anchors with e >= 0.12
"""
import sys, time
import numpy as np, torch
from common import *
from isobank import long_range_pairs, metric_guided_pairs, filter_by_distance

def guided_with_alpha(metric, anchors, r_target, n, fr, C, seed, valid_fn):
	"""metric_guided_pairs + the realised alpha (=sqrt(dθ g dθ)/r_target)."""
	g = metric.get_metric(torch.tensor(anchors), metric_type='hessian')
	t1, t2 = metric_guided_pairs(anchors, g, r_target, n, fr, C.lo, C.hi,
	                             seed=seed, phi_col=C.phi_col, valid_fn=valid_fn)
	with torch.no_grad():
		d = metric.proper_distance(t1, t2)
		ga = metric.get_metric(t1, metric_type='hessian')
		dt = (t2 - t1).double(); ga = torch.as_tensor(ga).double()
		alpha = torch.sqrt(torch.einsum('ni,nij,nj->n', dt, ga, dt)) / r_target
	keep = torch.isfinite(d) & (d > 0)
	return t1[keep], t2[keep], d[keep], alpha[keep].float()

def safe_metric(metric, pts, chunk=200):
	"""get_metric, dropping the rare anchors whose finite-difference stencil
	leaves the waveform domain (e within a few epsilon of 0 -> LAL domain error).
	Returns (pts_kept, g_kept)."""
	gs, keep = [], np.zeros(len(pts), bool)
	for s in range(0, len(pts), chunk):
		blk = pts[s:s+chunk]
		try:
			gs.append(np.asarray(metric.get_metric(torch.tensor(blk), metric_type='hessian')))
			keep[s:s+chunk] = True
		except (RuntimeError, ValueError):
			for j in range(len(blk)):
				try:
					gs.append(np.asarray(metric.get_metric(torch.tensor(blk[j:j+1]), metric_type='hessian')))
					keep[s+j] = True
				except (RuntimeError, ValueError):
					pass
	if (~keep).any():
		print(f'  safe_metric: dropped {(~keep).sum()} / {len(pts)} anchors (waveform domain error)')
	return pts[keep], np.concatenate(gs)

def main(which, p=0):
	"""p > 0: re-draw the training pairs with every seed offset by p (variance runs)."""
	C = load_config(); metric = C.make_metric(); vf = valid_fn_from(C)
	r_target = float(np.load(R_TARGET_FILE)[0])
	print(f'r_target = {r_target:.5f}'); sys.stdout.flush()

	if which in ('train', 'both'):
		t0 = time.time()
		train_data, val_data = C.sample_data(seed=p)
		O = 1000 * p
		long_p = long_range_pairs(train_data, metric, C.N_LONG, d_max=C.LONG_DMAX,
		                          seed=42 + O, phi_col=C.phi_col)
		train_data, g_tr = safe_metric(metric, train_data)
		val_data,   g_va = safe_metric(metric, val_data)
		lt1, lt2 = metric_guided_pairs(train_data, g_tr, r_target, C.N_LOCAL, C.LOCAL_FRACS,
		                               C.lo, C.hi, seed=123 + O, phi_col=C.phi_col, valid_fn=vf)
		dmax = C.LOCAL_DMAX_R * r_target if C.LOCAL_DMAX_R else np.inf
		local_p = filter_by_distance(metric, lt1, lt2, C.LOCAL_DMIN_R * r_target, dmax)
		bt1, bt2 = metric_guided_pairs(train_data, g_tr, r_target, C.N_BOUND, C.BOUND_FRACS,
		                               C.lo, C.hi, seed=555 + O, phi_col=C.phi_col, valid_fn=vf)
		bound_p = filter_by_distance(metric, bt1, bt2, 0.5 * r_target, 1.5 * r_target)
		ea = C.sample_edge_anchors(C.N_EDGE_ANCHORS, seed=777 + O)
		ea, g_e = safe_metric(metric, ea)
		et1, et2 = metric_guided_pairs(ea, g_e, r_target, C.N_EDGE, C.BOUND_FRACS,
		                               C.lo, C.hi, seed=778 + O, phi_col=C.phi_col, valid_fn=vf)
		edge_p = filter_by_distance(metric, et1, et2, 0.5 * r_target, 1.5 * r_target)
		mt1, mt2 = metric_guided_pairs(val_data, g_va, r_target, C.N_MON, C.BOUND_FRACS,
		                               C.lo, C.hi, seed=999 + O, phi_col=C.phi_col, valid_fn=vf)
		mon_p = filter_by_distance(metric, mt1, mt2, 0.5 * r_target, 1.5 * r_target)
		torch.save(dict(long=long_p, local=local_p, bound=bound_p, edge=edge_p,
		                monitor=mon_p, train_data=train_data, r_target=r_target, pair_seed=p), pairs_path(p))
		print('train pairs:', {k: len(v[2]) for k, v in
		      dict(long=long_p, local=local_p, bound=bound_p, edge=edge_p, monitor=mon_p).items()},
		      f'[{(time.time()-t0)/60:.1f} min]'); sys.stdout.flush()

	if which in ('test', 'both'):
		t0 = time.time()
		_, test_data = C.sample_data(n_train=10, n_val=6000, seed=2026)   # new anchors
		rng = np.random.default_rng(31)
		uni = C._sample_valid(C.lo, C.hi, 6000, rng)
		gl = long_range_pairs(uni, metric, 3000, d_max=None, seed=32, phi_col=C.phi_col)
		lo_ = guided_with_alpha(metric, test_data, r_target, 4000, (0.1, 2.5), C, 33, vf)
		bd_ = guided_with_alpha(metric, test_data, r_target, 4000, (0.6, 1.4), C, 34, vf)
		keep = (bd_[2] > 0.5 * r_target) & (bd_[2] < 1.5 * r_target)
		bd_ = tuple(x[keep] for x in bd_)
		ea = C.sample_edge_anchors(3000, seed=35)
		ed_ = guided_with_alpha(metric, ea, r_target, 3000, (0.6, 1.4), C, 36, vf)
		keep = (ed_[2] > 0.5 * r_target) & (ed_[2] < 1.5 * r_target)
		ed_ = tuple(x[keep] for x in ed_)
		torch.save(dict(glob=gl, local=lo_, bound=bd_, edge=ed_, r_target=r_target), TESTPAIRS)
		print('test pairs:', dict(glob=len(gl[2]), local=len(lo_[2]), bound=len(bd_[2]),
		      edge=len(ed_[2])), f'[{(time.time()-t0)/60:.1f} min]')

if __name__ == '__main__':
	main(sys.argv[1] if len(sys.argv) > 1 else 'both', int(parse_kv(sys.argv).get('p', 0)))
