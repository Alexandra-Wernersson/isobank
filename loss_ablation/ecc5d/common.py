"""
Shared settings for the multi-scale loss ablation on the 5D eccentric space.

Every arm uses the release package (projects/isobank), the shipped config
(configs/eccentric_5d.py), the shipped r_target, and one cached set of
training pairs, so the only thing that differs between arms is the loss.
"""
import os, sys, importlib.util

ISOBANK_ROOT = os.environ.get('ISOBANK_ROOT', '/home/awernersson/projects/isobank')   # the isobank release (configs/, runs/eccentric_5d/)
sys.path.insert(0, ISOBANK_ROOT)                 # release package, not isobank_testing's copy

HERE      = os.path.dirname(os.path.abspath(__file__))
RUNS      = os.path.join(HERE, 'runs')
PAIRS     = os.path.join(RUNS, 'train_pairs.pt')     # training pairs (shared by all arms)
TESTPAIRS = os.path.join(RUNS, 'test_pairs.pt')      # fresh held-out pairs for evaluation
CONFIG    = os.path.join(ISOBANK_ROOT, 'configs', 'eccentric_5d.py')
#	reference flow used ONLY to shortlist templates for coverage scoring
#	(identical for every arm, so retrieval quality is not confounded with the arm)
REF_FLOW  = os.path.join(ISOBANK_ROOT, 'runs', 'eccentric_5d', 'flow_best.pt')
R_TARGET_FILE = os.path.join(ISOBANK_ROOT, 'runs', 'eccentric_5d', 'r_target.npy')

#	arm -> (weights (w_glob, w_loc, w_bnd, w_edge), tail_w)
#	The paper recipe is (1, 5, 10, 10) with tail_w = 3 (config TAIL_W).
ARMS = {
	'full':        ((1.0, 5.0, 10.0, 10.0), 3.0),
	'no_global':   ((0.0, 5.0, 10.0, 10.0), 3.0),
	'no_local':    ((1.0, 0.0, 10.0, 10.0), 3.0),
	'no_bound':    ((1.0, 5.0,  0.0, 10.0), 3.0),
	'no_edge':     ((1.0, 5.0, 10.0,  0.0), 3.0),
	'global_only': ((1.0, 0.0,  0.0,  0.0), 3.0),
	'no_tail':     ((1.0, 5.0, 10.0, 10.0), 0.0),
}
ARM_ORDER = list(ARMS)
SEEDS = [0, 1]

#	Added later and kept out of ARMS, so existing array indices and tables do not shift.
EXTRA_ARMS = {
	'loc_edge':    ((0.0, 5.0,  0.0, 10.0), 3.0),      # only L_loc and L_edge
}
#	Weight sensitivity: one coefficient at a time, x0.5 and x2 (L_glob held at 1); tail_w 1 and 10 instead of 3.
WEIGHT_ARMS = {
	'loc_x0p5':  ((1.0,  2.5, 10.0, 10.0), 3.0),
	'loc_x2':    ((1.0, 10.0, 10.0, 10.0), 3.0),
	'bnd_x0p5':  ((1.0,  5.0,  5.0, 10.0), 3.0),
	'bnd_x2':    ((1.0,  5.0, 20.0, 10.0), 3.0),
	'edge_x0p5': ((1.0,  5.0, 10.0,  5.0), 3.0),
	'edge_x2':   ((1.0,  5.0, 10.0, 20.0), 3.0),
	'tail_1':    ((1.0,  5.0, 10.0, 10.0), 1.0),
	'tail_10':   ((1.0,  5.0, 10.0, 10.0), 10.0),
}
EXTRA_ARMS.update(WEIGHT_ARMS)

#	Retraining-variance runs: full recipe, training pairs re-drawn with pair
#	seed p (p = 0 is the shared ablation pair set), torch seed 100 + p.
VAR_PAIR_SEEDS = [1, 2, 3]

def pairs_path(p=0):
	return PAIRS if p == 0 else os.path.join(RUNS, f'train_pairs_p{p}.pt')

def run_name(arm, seed, p=0):
	return f'{arm}_s{seed}' if p == 0 else f'{arm}_p{p}_s{seed}'

def run_dir(arm, seed, p=0):
	d = os.path.join(RUNS, run_name(arm, seed, p))
	os.makedirs(d, exist_ok=True)
	return d

def load_config():
	spec = importlib.util.spec_from_file_location('case_config', CONFIG)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod

def valid_fn_from(C):
	return None if C.valid_mask is None else (lambda t: bool(C.valid_mask(t[None])[0]))

def parse_kv(argv):
	"""Optional key=value command-line arguments (ckpt=final, p=2, ...)."""
	return dict(a.split('=', 1) for a in argv if '=' in a)
