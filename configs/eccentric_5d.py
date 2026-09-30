"""
5D eccentric aligned-spin BNS/NSBH space (m1, m2, s1z, s2z, e).

  m1 in [1, 7] Msun, m2 in [1, 3] Msun,
  s1z in [-0.9, 0.9], s2z in [-0.05, 0.05], e in [1e-5, 0.15] @ 15 Hz
  TaylorF2Ecc, aligo_O3actual_H1 (ASD), f_min = 15 Hz
  (the Phukon, Schmidt & Pratten 2412.06433 BNS/NSBH space)


VALIDITY: m1 >= m2 by convention; enforced in all sampling.

Validated result: flow monitor p10 = 0.96 with this recipe.
"""

import os
import numpy as np

from isobank import cbc_metric, load_PSD, get_boundaries_from_ranges

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('ISOBANK_ROOT') or os.path.dirname(HERE)   # repository root
#	set ISOBANK_ROOT if this config lives outside the repository
OUT_DIR = os.path.join(ROOT, 'runs', 'eccentric_5d')      # checkpoints, banks, radii
os.makedirs(OUT_DIR, exist_ok=True)
PSD_FILE = os.path.join(ROOT, 'data', 'psds', 'aligo_O3actual_H1.txt')

variable_format = 'm1m2_s1z_s2z_e'
boundaries = get_boundaries_from_ranges(variable_format,
        (1, 7), (1, 3),
        s1_range=(-0.9, 0.9),
        s2_range=(-0.05, 0.05),
        e_range=(1e-5, 0.15))
lo, hi = np.array(boundaries[0]), np.array(boundaries[1])
D = 5
M1_COL, M2_COL, E_COL = 0, 1, 4

def make_metric():
	return cbc_metric(variable_format,
	          PSD=load_PSD(PSD_FILE, True, 'H1'),   # ASD file
	          approx='TaylorF2Ecc', f_min=15, f_max=1024)

#	Angles / lifting: none
phi_col  = None
embed_fn = None
d_flow   = D

#	Validity: m1 >= m2
def valid_mask(batch):
	batch = np.asarray(batch)
	return batch[:, M1_COL] >= batch[:, M2_COL]

#	Flow architecture — 3-cycle masks: every dimension transformed at least
#	once per cycle and conditioned on in at least two patterns.
#	DEPTH=1 (Linear-Tanh-Linear conditioners) and TAIL_W=3.0 are the values
#	of the validated original run (monitor p10 = 0.96).
n_layers, hidden_dim = 9, 256
DEPTH  = 1
TAIL_W = 3.0
masks = [[1, 1, 0, 0, 0],   # condition (m1,m2),   transform (s1z,s2z,e)
         [0, 0, 1, 1, 0],   # condition (s1z,s2z), transform (m1,m2,e)
         [1, 0, 1, 0, 1]]   # condition (m1,s1z,e), transform (m2,s2z)

#	Training data — tiered eccentricity mix, m1 >= m2 enforced
def _sample_valid(lo_, hi_, n, rng):
	out = []
	n_have = 0
	while n_have < n:
		batch = rng.uniform(lo_, hi_, (max(n - n_have, 256) * 2, D))
		batch = batch[valid_mask(batch)]
		out.append(batch); n_have += len(batch)
	return np.vstack(out)[:n].astype(np.float32)

def sample_data(n_train=10000, n_val=4000, seed=0):
	rng = np.random.default_rng(seed)
	e_lo, e_hi = lo[E_COL], hi[E_COL]
	e_sub1 = e_lo + 0.6 * (e_hi - e_lo)
	e_edge = e_lo + 0.8 * (e_hi - e_lo)
	def tiered(n):
		n_u, n_1 = int(0.3 * n), int(0.3 * n)
		n_2 = n - n_u - n_1
		lo1 = lo.copy(); lo1[E_COL] = e_sub1
		lo2 = lo.copy(); lo2[E_COL] = e_edge
		data = np.vstack([_sample_valid(lo,  hi, n_u, rng),
		                  _sample_valid(lo1, hi, n_1, rng),
		                  _sample_valid(lo2, hi, n_2, rng)])
		rng.shuffle(data)
		return data
	return tiered(n_train), tiered(n_val)

#	Pair-generation settings (matching the validated original run).
#	LONG_DMAX=None: this space is hundreds of proper-distance units across;
#	the long-range pairs must span the full distance range or the flow never
#	learns the global scale and the ratio losses stay clamped (dead) at init.
#	Local pairs span fracs up to 2.5 with no upper distance filter;
#	boundary/edge pairs concentrate tightly around placement scale (0.6-1.4).
N_LONG,  LONG_DMAX             = 10000, None
N_LOCAL, LOCAL_FRACS, LOCAL_DMIN_R = 5000, (0.1, 2.5), 0.0
LOCAL_DMAX_R                   = None
N_BOUND, BOUND_FRACS           = 10000, (0.6, 1.4)
N_MON                          = 3000

#	Dedicated EDGE pairs: anchors uniform in the high-e region (e >= 80% of
#	range). Adds the L_edge loss (weight 10) and switches checkpointing to
#	max min(p10_monitor, p10_edge) — without this the flow converges ~10
#	p10-points lower (validated: 0.86 without vs 0.96 with).
N_EDGE_ANCHORS, N_EDGE = 1500, 5000

def sample_edge_anchors(n, seed):
	rng = np.random.default_rng(seed)
	e_lo, e_hi = lo[E_COL], hi[E_COL]
	edge_lo = lo.copy(); edge_lo[E_COL] = e_lo + 0.8 * (e_hi - e_lo)
	return _sample_valid(edge_lo, hi, n, rng)

#	Dedicated DEGENERATE-DIRECTION pairs: step along the smallest-eigenvalue
#	eigenvectors of the local metric. A random-direction step never samples
#	these, so the flow over-expands them and greedy placement keeps redundant
#	near-duplicate templates (measured: median nearest-neighbour proper
#	distance 0.68 r_target -> ~2x too many templates; the redundant pairs
#	differ along a tilted mass-plane eigenvector). These pairs, trained with
#	the ratio loss, teach the flow to collapse those directions.
N_DEGEN         = 8000
N_SOFT          = 2               # target the 2 smallest-eigenvalue directions
DEGEN_MAG_RANGE = (0.1, 0.9)      # step as a fraction of distance-to-boundary
W_DEGEN         = 10.0
DEGEN_BEST_CKPT  = os.path.join(OUT_DIR, 'flow_best_degen.pt')
DEGEN_FINAL_CKPT = os.path.join(OUT_DIR, 'flow_final_degen.pt')

#	Training
N_EPOCHS = 1000

#	Calibration
MU, N_REF = 0.03, 400

#	Placement: stop when a batch of candidates is >90% covered. Calibrated
#	from the measured size-coverage curve (size_coverage_curve.py): batch
#	acceptance ~0.10 corresponds to ~2.5M templates and 97.5% true-match
#	coverage at FF>=0.97 — the standard search target, with a small margin.
#	The bank grows steeply for the last few points of coverage
#	(0.01 acceptance -> 8.2M templates / 99.7%), so tighten only for a
#	higher coverage target.
STOP_ACCEPTANCE = 0.10

#	Coverage
N_INJ, K_NEIGHBOURS = 2000, 20

#	Output files (in this folder)
R_TARGET_FILE = os.path.join(OUT_DIR, 'r_target.npy')
BEST_CKPT     = os.path.join(OUT_DIR, 'flow_best.pt')
FINAL_CKPT    = os.path.join(OUT_DIR, 'flow_final.pt')
TEMPLATE_FILE = os.path.join(OUT_DIR, 'templates.npy')

#	Plot config (m1 is linear in this format)
mass_col, log_mass, y_col, y_label = 0, False, 4, 'e'
CASE_NAME = 'Eccentric 5D BNS/NSBH (m1, m2, s1z, s2z, e)'

#	Axis labels for the figure scripts (one per coordinate)
PARAM_LABELS = [r'$m_1\ [M_\odot]$', r'$m_2\ [M_\odot]$', r'$\chi_1$', r'$\chi_2$', r'$e$']
