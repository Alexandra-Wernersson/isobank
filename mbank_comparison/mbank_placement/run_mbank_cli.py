"""Compat wrapper for mbank CLI scripts against a newer scientific stack.
Usage: python run_mbank_cli.py bin/<cli> <args...>

Shims applied:
  * numpy>=1.24 removed np.float (old emcee references it)
  * emcee>=3.1 dropped the progress_kwargs argument to sample()/run_mcmc()
"""
import sys
import numpy as np
if not hasattr(np, 'float'): np.float = float

import emcee
_orig_sample = emcee.EnsembleSampler.sample
def _sample(self, *a, **k):
    k.pop('progress_kwargs', None)
    return _orig_sample(self, *a, **k)
emcee.EnsembleSampler.sample = _sample

path = sys.argv.pop(1)
exec(compile(open(path).read(), path, 'exec'))
