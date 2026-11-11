"""
place_al3d_mbank_stochastic.py
--------------------------------
mbank's STOCHASTIC placement (place_stochastically_flow) on the aligned-spin
3D space — the comparison that is infeasible in 5D but tractable here.

This is the like-for-like test against isobank: both methods are stochastic
(propose / reject if too close), both target the same maximum mismatch. The
ONE difference is the distance used in the acceptance test:

  mbank   : metric distance, requiring get_metric() at EVERY proposal
  isobank : Euclidean distance in the flow's latent space, no metric calls

Proposals are drawn from mbank's own density flow when it can reach the
space; a uniform fallback is used otherwise (the placement algorithm is
agnostic to the proposal distribution, and using the same stream for both
methods keeps the comparison controlled).

Records wall time and the metric-evaluation count so the cost can be
compared directly against isobank's placement.
"""

import os, sys, time
import numpy as np
import torch

from mbank import cbc_metric
from mbank.utils import load_PSD, get_boundaries_from_ranges
from mbank.placement import place_stochastically_flow
from mbank.flow import STD_GW_Flow

MM       = float(os.environ.get('AL3D_MM', 0.97))     # minimal-match sweep: AL3D_MM=0.96 etc.
OUT      = 'bank_al3d_mbank_stochastic.npy' if MM == 0.97 else f'bank_al3d_mbank_stochastic_mm{MM:g}.npy'
EMPTY_IT = 2000          # consecutive rejections before declaring saturation
variable_format = 'Mq_chi'

boundaries = get_boundaries_from_ranges(variable_format,
        (30, 50), (1, 5), chi_range=(-0.99, 0.99))
lo, hi = np.array(boundaries[0]), np.array(boundaries[1])
D = 3

metric_obj = cbc_metric(variable_format,
            PSD=load_PSD('aLIGOZeroDetHighPower.txt', False, 'H1'),
            approx='IMRPhenomD', f_min=10, f_max=1024)

def boundaries_checker(theta):
    theta = np.atleast_2d(theta)
    return np.all((theta >= lo) & (theta <= hi), axis=1)

# ── Proposal sampler ──────────────────────────────────────────────────────────
FLOW_FILE = 'al3d_mbank_native/flow_al3d_native.zip'

class UniformSampler:
    """Fallback proposal sampler: uniform over the box. The stochastic
    placement algorithm only needs a stream of proposals."""
    def __init__(self, lo, hi, seed=0):
        self.lo, self.hi = lo, hi
        self.rng = np.random.default_rng(seed)
    def sample(self, n):
        return torch.tensor(self.rng.uniform(self.lo, self.hi, (int(n), len(self.lo))),
                            dtype=torch.float32)
    def sample_within_boundaries(self, n, checker=None):
        s = self.sample(n)
        return s, torch.zeros(len(s))

if os.path.exists(FLOW_FILE):
    flow = STD_GW_Flow.load_flow(FLOW_FILE)
    with torch.no_grad():
        probe = flow.sample(20_000).numpy()
    frac_in = boundaries_checker(probe).mean()
    print(f"mbank flow: {100*frac_in:.1f}% of samples land inside the space")
    if frac_in > 0.05:
        sampler = flow
        print("  -> using the mbank density flow as the proposal sampler")
    else:
        sampler = UniformSampler(lo, hi)
        print("  -> flow rarely reaches the space; using UNIFORM proposals")
else:
    sampler = UniformSampler(lo, hi)
    print(f"{FLOW_FILE} not found -> using UNIFORM proposals")
sys.stdout.flush()

# ── Count metric evaluations ──────────────────────────────────────────────────
_orig_get_metric = metric_obj.get_metric
_counter = {'n': 0}
def counting_get_metric(*a, **k):
    _counter['n'] += 1
    return _orig_get_metric(*a, **k)
metric_obj.get_metric = counting_get_metric

print(f"\nmbank STOCHASTIC placement: MM={MM}, empty_iterations={EMPTY_IT}")
sys.stdout.flush()
t0 = time.time()
templates = place_stochastically_flow(MM, sampler, metric_obj,
                                      boundaries_checker=boundaries_checker,
                                      empty_iterations=EMPTY_IT, verbose=True)
elapsed = time.time() - t0
templates = np.asarray(templates, dtype=np.float32)
np.save(OUT, templates)

print(f"\n{'='*64}")
print(f"mbank stochastic placement — aligned 3D")
print(f"{'='*64}")
print(f"  templates            : {len(templates):,}")
print(f"  wall time            : {elapsed/60:.1f} min")
print(f"  metric evaluations   : {_counter['n']:,}")
print(f"  ms per metric eval   : {1000*elapsed/max(_counter['n'],1):.1f}")
print(f"\n  isobank stochastic placement of the same space required")
print(f"  ZERO metric evaluations (23,810 templates).")
print(f"{'='*64}")
print("Saved:", OUT)
