"""
FF cumulative distributions, isobank vs mbank, both parameter spaces.

Colour = parameter space (blue eccentric 5D, orange aligned-spin 3D).
Line style = pipeline (solid isobank, dashed mbank).

All four curves use a k=500 latent-space shortlist, verified converged, and the
shortlist is provably complete at the 0.97 threshold: no pair of waveforms in
these spaces reaches match >= 0.97 beyond 1.85 r_target, while the shortlist
extends to at least 4.1 r_target for every injection and both banks.

  eccentric 5D    : 10,000 injections, uniform in coordinates, seed 4002.
                    isobank 2,267,558 vs mbank's FULL native bank 2,953,739
                    (the abstract's 23%-fewer-templates comparison).
  aligned-spin 3D : 10,000 injections, seed 4242, k=100 (flat from k=20).
                    isobank 23,810 templates vs
                    mbank stochastic 23,167 (like-for-like placement; mbank's
                    native `random` placement gives 19,767 / 94.85%).
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
AL   = os.path.join(HERE, '..', 'aligned_3d')
os.chdir(HERE)
THR = 0.97
C_ECC, C_AL = '#0284c7', '#d97706'

CURVES = [
    ('cell_ALL_k500_ff.npy',                             C_ECC, '-',            2_267_558, 'isobank', 'eccentric 5D'),
    ('ff_mbank_full_k500_seed4002.npy',                  C_ECC, (0, (5, 2)),    2_953_739, 'mbank',   'eccentric 5D'),
    (os.path.join(AL, 'ff_al3d_isobank_shipped_k100_N10k_seed4242.npy'),
                                                          C_AL,  '-',             23_810, 'isobank', 'aligned-spin 3D'),
    (os.path.join(AL, 'ff_al3d_mbank_stochastic_k100_N10k_seed4242.npy'),
                                                          C_AL, (0, (5, 2)),      23_167, 'mbank',   'aligned-spin 3D'),
]

fig, ax = plt.subplots(figsize=(6.4, 4.6))
rows = []
for path, col, ls, n, pipe, space in CURVES:
    ff = np.load(path)
    s = np.sort(ff); y = 100*np.arange(1, len(s)+1)/len(s)
    cov = 100*(ff >= THR).mean(); se = 100*np.sqrt((cov/100)*(1-cov/100)/len(ff))
    ax.semilogy(s, y, color=col, ls=ls, lw=2.0, zorder=4 if col == C_ECC else 3,   # eccentric (blue) drawn above aligned (orange)
                label=f'{pipe} · {space} ({n:,}) — {cov:.2f}%',
                solid_capstyle='round', dash_capstyle='round')
    rows.append((pipe, space, n, cov, se, np.percentile(ff, 10), ff.min(), len(ff)))
ax.axvline(THR, color='#dc2626', ls=':', lw=1.6, zorder=5)   # FF = 0.97 threshold, red
ax.set_xlabel('Fitting factor', fontsize=11)
ax.set_ylabel('Cumulative fraction of injections (%)', fontsize=11)
ax.set_xlim(0.85, 1.0); ax.set_ylim(0.008, 150)
ax.tick_params(axis='both', labelsize=9)
ax.grid(True, which='both', alpha=0.25)
ax.legend(fontsize=8.2, loc='upper left', framealpha=0.92)
plt.tight_layout()
plt.savefig('ff_cdf_isobank_vs_mbank.pdf', dpi=150, bbox_inches='tight', facecolor='white')
print("saved ff_cdf_isobank_vs_mbank.pdf\n")
print(f"{'pipeline':>8} | {'space':>16} | {'templates':>10} | {'coverage':>16} | {'p10 FF':>7} | {'min FF':>7} | N_inj")
for pipe, space, n, cov, se, p10, mn, N in rows:
    print(f"{pipe:>8} | {space:>16} | {n:>10,} | {cov:6.2f}% +/- {se:.2f} | {p10:7.4f} | {mn:7.4f} | {N:,}")
