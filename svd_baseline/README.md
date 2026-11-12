# SVD-lattice baseline for isobank (comparison with Cabass et al., arXiv:2511.10466)

Cabass et al. build template banks from the top SVD coefficients of the PSD-weighted waveform phase: an autoencoder maps
the coefficients to a low-dimensional, distance-preserving latent space, and templates are placed on a lattice there.
Their released code (github.com/giovanni-cabass/autoencoder_GWs) contains only the autoencoder and its loss.

These scripts test the representation the autoencoder is built on, **without any autoencoder**: a lattice placed
directly in the phase-SVD coefficients. For waveforms that share an amplitude profile, Euclidean distance between
coefficient vectors approximates the mismatch, 1 - M ≈ |Δc|²/2, so a lattice in coefficient space is a template bank
(each lattice point is a waveform rebuilt from its coefficients). An autoencoder is needed when the phases need many more
coefficients than the space has dimensions; on the eccentric space, 3 coefficients already suffice.

Spaces (the two of the isobank paper):

| space | parameters | model, PSD, band | injections |
|---|---|---|---|
| `ecc5d` | m1 1–7, m2 1–3, s1z ±0.9, s2z ±0.05, e 1e-5–0.15 | TaylorF2Ecc, O3 H1 ASD, 15–1024 Hz | first 4,000 of seed 4002 |
| `al3d` | M 30–50, q 1–5, χ ±0.99 (equal spins) | IMRPhenomD, aLIGOZeroDetHighPower PSD, 10–1024 Hz | first 5,000 of seed 4242 |

The injections are the same draws as the isobank and mbank coverage numbers in the paper.

## Requirements

numpy, scipy, torch, lalsuite (`lal`, `lalsimulation`); the versions we used are in `isobank/environment-pinned.yml`.

## Steps

Submit each job from this folder (`sbatch job_....sh`; array index 0 = ecc5d, 1 = al3d). Outputs go to `results/`.

| step | script | job | what it does | output |
|---|---|---|---|---|
| 1 | `svd_feasibility.py <space>` | `job_feasibility.sh` | SVD of 2,500 waveforms' phases; overlap of 800 held-out waveforms with their reconstruction from k = 2…200 components; spread of the amplitude profile | `svd_feasibility_<space>.json` |
| 2 | `svd_grid_estimate.py <space>` | `job_grid_estimate.sh` | coefficients of 6,000 training waveforms, box corners and the injections; exact coverage of a cubic lattice; bank size from a convex-hull upper bound (not available for ecc5d, where the hull fails) | `svd_coeffs_<space>.npz` (needed by step 3), `svd_grid_<space>.json` |
| 3 | `svd_grid_occupancy.py <space>` | `job_grid_occupancy.sh` | bank size = number of lattice cells reached by the physical space, via an MLP regressor parameters → coefficients sampled up to 1e8 times; small-angle coverage | `svd_occupancy_<space>.json` |
| 4 | `svd_grid_refined.py ecc5d` | `job_grid_refined.sh` | final 5D numbers: exact phase overlap, regressor with 20,000 extra low-mass waveforms, cells sampled uniformly in Mc^(-5/3) up to 2e8 (converged to ~2%), cubic and body-centred cubic (BCC) lattices. Independent of steps 2–3 | `svd_refined_ecc5d.json` |

`svd_grid_estimate.py` and `svd_grid_refined.py` import `svd_feasibility.py` for the waveforms and settings.
The `.npz` coefficient files (about 1 MB each) are not tracked; step 2 recreates them.
Rerunning a step overwrites our results in `results/`; all draws are seeded, and step 1 for al3d reproduces our file exactly.

Run times (32-core node): step 1 a few minutes; step 2 about 10 minutes (ecc5d); step 3 up to 3 h; step 4 about 4 h.

## Our results (in `results/`)

5D, same 4,000 injections, coverage at FF ≥ 0.97:

| bank | templates | coverage | Mc < 1.4 | lowest FF |
|---|---|---|---|---|
| SVD, BCC lattice, spacing 0.48 | 2.61M | 99.12% | 94.2% | 0.902 |
| SVD, BCC lattice, spacing 0.52 | 2.08M | 94.33% | 91.3% | 0.874 |
| isobank, paper bank | 2.27M | 96.75% | 94.7% | 0.935 |
| mbank, mm 0.978 | 4.84M | 96.08% | 73.4% | 0.835 |

(isobank and mbank values come from their saved fitting factors on the same injections, not from these scripts.)

3D (`svd_occupancy_al3d.json`, 3 coefficients, cubic spacing 0.283): 21,283 cells; 98.6% coverage with phase only,
93.1% with the real amplitudes (one amplitude profile), against isobank's 23,810 templates at 99.7%.

## Caveats

- Templates are points in coefficient space (reconstructed phases with one amplitude profile), not physical parameters.
- Coverage is the phase overlap with the snapped template, without further time/phase maximisation (slightly pessimistic).
- Bank sizes rely on the regressor; its rms error on the first coefficient (0.29) is below the lattice spacing.
- `svd_occupancy_ecc5d.json` (step 3) is not converged for 5D; use `svd_refined_ecc5d.json` (step 4).
- The autoencoder itself is not run.
