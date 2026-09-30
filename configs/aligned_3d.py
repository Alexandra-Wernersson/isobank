"""
Case config — 3D aligned-spin BBH space (M, q, chi_eff).

  M   in [30, 50] Msun
  q   in [1, 5]
  chi in [-0.99, 0.99]
  IMRPhenomD, aLIGOZeroDetHighPower, f_min = 10 Hz

PURPOSE: proof of principle. This is the regime where classical geometric /
lattice placement already works.

Mass format is 'Mq', so M is LINEAR (not log) in the coordinates.
"""

import os
import numpy as np

from isobank import cbc_metric, load_PSD, get_boundaries_from_ranges

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get('ISOBANK_ROOT') or os.path.dirname(HERE)   # repository root
#	set ISOBANK_ROOT if this config lives outside the repository
OUT_DIR = os.path.join(ROOT, 'runs', 'aligned_3d')      # checkpoints, banks, radii
os.makedirs(OUT_DIR, exist_ok=True)
PSD_FILE = os.path.join(ROOT, 'data', 'psds', 'aLIGOZeroDetHighPower.txt')

variable_format = 'Mq_chi'
boundaries = get_boundaries_from_ranges(variable_format,
        (30, 50), (1, 5), chi_range=(-0.99, 0.99))
lo, hi = np.array(boundaries[0]), np.array(boundaries[1])
D = 3
M_COL, Q_COL, CHI_COL = 0, 1, 2

def make_metric():
	return cbc_metric(variable_format,
	          PSD=load_PSD(PSD_FILE, False, 'H1'),   # PSD file (not ASD)
	          approx='IMRPhenomD', f_min=10, f_max=1024)

#	Angles / lifting: none
phi_col  = None
embed_fn = None
d_flow   = D

#	Validity: the whole box is valid (Mq format with q>=1 already orders masses)
valid_mask = None

#	Flow architecture. D=3 default masks: [1,1,0] / [0,1,1] / [1,0,1] — every
#	coordinate transformed at least once and conditioned on twice per cycle.
#	Modest net: the lesson from the other cases is that the training objective,
#	not capacity, is the limiting factor.
n_layers, hidden_dim = 8, 128
DEPTH  = 2
TAIL_W = 3.0
masks  = None            # -> isobank.flow.default_masks(3)

#	Training data — uniform over the box (no known hard sub-region here; add a
#	tiered mix / edge set only if the monitor p10 shows a weak corner).
def sample_data(n_train=10000, n_val=4000, seed=0):
	rng = np.random.default_rng(seed)
	data = rng.uniform(lo, hi, (n_train + n_val, D)).astype(np.float32)
	return data[:n_train], data[n_train:]

#	Pair generation. LONG_DMAX=None: never truncate the long-range pairs —
#	they must span the full proper-distance range or the flow cannot learn the
#	global scale and the ratio losses stay clamped at initialisation.
N_LONG,  LONG_DMAX                 = 10000, None
N_LOCAL, LOCAL_FRACS, LOCAL_DMIN_R = 5000, (0.1, 2.5), 0.0
LOCAL_DMAX_R                       = None
N_BOUND, BOUND_FRACS               = 10000, (0.6, 1.4)
N_MON                              = 3000

#	Training
N_EPOCHS = 1000

#	Calibration
MU, N_REF = 0.03, 400

#	Placement: run near saturation (a 3D bank is cheap), then truncate to the
#	smallest prefix meeting the coverage target using the measured
#	size-coverage curve. The acceptance->coverage mapping is space-specific
#	and must NOT be inherited from another case.
STOP_ACCEPTANCE = 0.01

#	Coverage
N_INJ, K_NEIGHBOURS = 3000, 10

#	Output files (in this folder)
R_TARGET_FILE = os.path.join(OUT_DIR, 'r_target.npy')
BEST_CKPT     = os.path.join(OUT_DIR, 'flow_best.pt')
FINAL_CKPT    = os.path.join(OUT_DIR, 'flow_final.pt')
TEMPLATE_FILE = os.path.join(OUT_DIR, 'templates.npy')

#	Plot config: M is linear in the Mq format; show the (M, chi) plane
mass_col, log_mass, y_col, y_label = 0, False, 2, r'$\chi_{\rm eff}$'
CASE_NAME = 'Aligned-spin 3D BBH (M, q, chi)'

#	Axis labels for the figure scripts (one per coordinate)
PARAM_LABELS = [r'$M\ [M_\odot]$', r'$q$', r'$\chi$']
