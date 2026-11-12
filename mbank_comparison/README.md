# isobank vs mbank: the comparison in the paper (Fig. 3, Table FF)

Everything behind the isobank–mbank comparison in `mbank/iclr2026_conference.tex` (Sec. 6.3 and Fig. 3), gathered in one place.

| space | bank | templates | coverage (FF ≥ 0.97) |
|---|---|---|---|
| eccentric 5D | isobank (`templates_matched_2p27M.npy`) | 2,267,558 | 96.64 ± 0.18% |
| eccentric 5D | mbank, random placement, mm 0.97, cf 0.9 (`bank_ecc5D_native.dat`) | 2,953,739 | 94.17 ± 0.23% |
| aligned-spin 3D | isobank (`templates.npy`) | 23,810 | 99.72 ± 0.05% |
| aligned-spin 3D | mbank, stochastic placement, mm 0.97 (`bank_al3d_mbank_stochastic.npy`) | 23,167 | 99.80 ± 0.04% |

## Protocol (identical for both codes)

- **5D eccentric** (TaylorF2Ecc, O3 H1 ASD, 15–1024 Hz): 10,000 injections uniform in (m1, m2, s1z, s2z, e) with m1 ≥ m2, seed 4002.
  For each injection, the 500 nearest templates in isobank's latent space (`ecc5d/flow_best.pt`) are shortlisted and the
  exact match is computed with each; the fitting factor is the largest.
- **3D aligned spin** (IMRPhenomD, aLIGOZeroDetHighPower PSD, 10–1024 Hz): 10,000 injections uniform in (M, q, χ), seed 4242,
  100 nearest templates (`al3d/flow_best.pt`); converged from k = 20.
- The shortlist is complete at the 0.97 threshold: in sampled pairs, no two waveforms reach match 0.97 beyond 1.85 r_target,
  and the shortlist reaches at least 2.9 r_target for every injection (an empirical check, not a proof).
- Note: the paper text says both spaces use the O3 PSD; the 3D space actually uses aLIGOZeroDetHighPower for both codes.

## Contents

- `mbank_placement/`: how the mbank banks were made.
  - 5D: `job_ecc5D_mbank_native.sh` runs mbank's command-line tools (dataset, flow training, placement) with `ecc5D_mbank_native.ini`.
  - 3D: `job_al3d_mbank_native.sh` trains the flow with `al3d_mbank_native.ini`; `job_al3d_mbank_stoch.sh` then runs
    `place_al3d_mbank_stochastic.py` (stochastic placement, flow proposals, 2,000 empty iterations).
    This copy of the script also accepts `AL3D_MM` for the minimal-match sweep; without it, it runs mm 0.97 as in the paper.
  - the trained mbank flows (`flow_*.zip`), the two PSDs and `run_mbank_cli.py` (compatibility wrapper for newer numpy/emcee).
  - The flow-training settings exceed mbank's defaults (15,000 points, 60 hidden features, 30,000 epochs, learning rate 0.005,
    10,000 livepoints); Sec. 6.3 says "default settings" and should be corrected.
- `ecc5d/`: `ff_cell_k500.py ALL` scores the isobank bank (→ `cell_ALL_k500_ff.npy`); `ff_mbank_full_k500.py` scores the mbank bank
  (→ `ff_mbank_full_k500_seed4002.npy`).
- `al3d/`: `al3d_ff_10k.py` scores both 3D banks (→ `ff_al3d_isobank_shipped_k100_N10k_seed4242.npy`, `ff_al3d_mbank_stochastic_k100_N10k_seed4242.npy`).
- `figure/`: `ff_cdf_isobank_vs_mbank.py` reads the four fitting-factor files and makes Fig. 3 (`ff_cdf_isobank_vs_mbank.pdf`).
- `tuning/`: mbank retrained with changes to its loss and training, with the paper's placement; HTCondor DAG for Nikhef
  (see [`tuning/README.md`](tuning/README.md)).

## Data files (not in this folder)

Banks and fitting-factor arrays are left out because of their size. On Snellius they are at:

| file | location |
|---|---|
| isobank 5D bank `templates_matched_2p27M.npy` (45 MB) | `isobank_testing/examples/eccentric_5d/` |
| mbank 5D bank `bank_ecc5D_native.dat` (372 MB) | `mbank/ecc5D_mbank_native/` |
| isobank 3D bank `templates.npy` | `isobank_testing/examples/aligned_3d/` |
| mbank 3D bank `bank_al3d_mbank_stochastic.npy` | `mbank/` |
| 5D fitting factors `cell_ALL_k500_ff.npy`, `ff_mbank_full_k500_seed4002.npy` | `isobank_testing/examples/eccentric_5d/` |
| 3D fitting factors `ff_al3d_*_k100_N10k_seed4242.npy` | `isobank_testing/examples/aligned_3d/` |

(paths relative to `/gpfs/home5/awernersson/projects/`)

## Running

The scripts are unmodified copies of the ones that produced the paper's numbers, and they keep their original paths:
they `chdir` to, and import `config.py` from, `isobank_testing/examples/eccentric_5d/` and `.../aligned_3d/`, and import
`isobank` from `isobank_testing/`. To rerun anything, run the originals in those folders (same file names).
The fitting-factor files listed above (80 kB each) are enough to regenerate the table and figure without rerunning any scoring.

The minimal-match sweep for the reviewer response, packaged to run anywhere, is in `mbank/mm_sweep/`.
