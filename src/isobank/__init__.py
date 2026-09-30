"""
isobank
=======

Template-bank generation with isometric normalizing flows.

A normalizing flow is trained so that Euclidean distance in its latent space
matches the proper distance of the waveform-match metric. Template placement
then reduces to ball packing in the latent space, and nearest-template lookups
to Euclidean nearest-neighbour queries.

Pipeline (see ``scripts/`` for runnable entry points):

1. **Calibrate** the placement radius ``r_target`` against TRUE matches
   (:mod:`isobank.metric.calibration`)
2. **Generate pairs** — long-range, metric-guided local and boundary pairs
   (:mod:`isobank.flow.pairs`)
3. **Train** the isometric flow (:mod:`isobank.flow.training`)
4. **Place** templates greedily in the latent space
   (:mod:`isobank.placement.placement`)
5. **Validate** with TRUE-match coverage (:mod:`isobank.evaluation.coverage`)

Names are resolved lazily, so the flow and placement code can be imported
without a waveform library installed; ``lalsuite`` is required only when the
metric is actually used.

The metric computation (:mod:`isobank.metric.metric`) and the binary
parametrization handling (:mod:`isobank.metric.handlers`) are adapted from
`mbank <https://github.com/stefanoschmidt1995/mbank>`_ (GPLv3) — see LICENSE.
"""

import importlib

__version__ = '0.1.0'

#	public name -> module providing it
_EXPORTS = {
    'cbc_metric':              'isobank.metric.metric',
    'variable_handler':        'isobank.metric.handlers',
    'load_PSD':                'isobank.metric.utils',
    'get_boundaries_from_ranges': 'isobank.metric.utils',
    'calibrate_r_target':      'isobank.metric.calibration',
    'IsometricFlow':           'isobank.flow.flow',
    'AffineCouplingLayer':     'isobank.flow.flow',
    'default_masks':           'isobank.flow.flow',
    'wrap_phi':                'isobank.flow.flow',
    'wrap_phi_np':             'isobank.flow.flow',
    'lift_sphere':             'isobank.flow.flow',
    'save_checkpoint':         'isobank.flow.flow',
    'load_checkpoint':         'isobank.flow.flow',
    'long_range_pairs':        'isobank.flow.pairs',
    'metric_guided_pairs':     'isobank.flow.pairs',
    'filter_by_distance':      'isobank.flow.pairs',
    'degenerate_direction_pairs': 'isobank.flow.pairs',
    'ratio_loss_with_tail':    'isobank.flow.losses',
    'log_distance_loss':       'isobank.flow.losses',
    'scale_percentiles':       'isobank.flow.losses',
    'train_isometric_flow':    'isobank.flow.training',
    'greedy_placement':        'isobank.placement.placement',
    'coverage_check':          'isobank.evaluation.coverage',
}

__all__ = sorted(_EXPORTS) + ['__version__']


def __getattr__(name):
    if name in _EXPORTS:
        value = getattr(importlib.import_module(_EXPORTS[name]), name)
        globals()[name] = value          # cache, so later lookups are free
        return value
    raise AttributeError(f"module 'isobank' has no attribute '{name}'")


def __dir__():
    return __all__
