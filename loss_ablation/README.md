# isobank: retraining variance, loss ablation and weight scan

Code and results behind the ablation section of the isobank paper (multi-scale loss, Eq. 7):

- **Retraining variance**: the full objective retrained five times on the eccentric space (two new initialization
  seeds on the shared training pairs, three with independently redrawn pairs), and four times on the aligned-spin space.
- **Ablation**: one term removed at a time (`no_global`, `no_local`, `no_bound`, `no_edge`, `no_tail`), only
  `L_loc` + `L_edge`, and `L_glob` alone.
- **Weight scan**: each local coefficient ×0.5 and ×2, tail weight 1 and 10 (nominal 3).

Banks are compared **at equal size**: for each trained flow the placement radius is searched until the bank has
2,267,558 templates (±3%) in 5D or 23,810 (±3%) in 3D, because the bank size at a fixed radius differs strongly
between trainings. Coverage uses exact matches on fixed injections (5D: 4,000, seed 4002, 200 nearest templates;
3D: 5,000, seed 4242, 20 nearest templates).

## Results

`python summary.py` prints every number of the ablation section from `results/` (one JSON per trained flow:
`matched_final.json` in 5D with miss rate, radius, bank size and miss rate by chirp mass; `history.json` with the training
curves; `matched.json` in 3D). Values ending in exactly 5 in the third decimal (e.g. 3.725%) are rounded up in the paper.

## Requirements

The isobank release (`ISOBANK_ROOT`, default `/home/awernersson/projects/isobank`), which provides the package, the
eccentric config (`configs/eccentric_5d.py`) and the paper's eccentric flow and r_target (`runs/eccentric_5d/`), used
to shortlist templates identically for every run. The 3D script also needs the paper's aligned-spin config, flow and
r_target (`AL3D_PAPER_DIR`). Package versions: `isobank/environment-pinned.yml`.

## Running (from each subfolder; job scripts are SLURM)

**Eccentric 5D** (`ecc5d/`; runs are written to `ecc5d/runs/`):

| order | job | what it does |
|---|---|---|
| 1 | `job_pairs.sh` | shared training pairs and held-out test pairs (~1 h on one core) |
| 2 | `job_retrain_pairs.sh` | three independently redrawn training-pair sets (34 min each) |
| 3 | `job_retrain_full.sh` | **retraining variance**: the full objective, 5 runs (~1.6 h each) |
| 3 | `job_train_arms.sh` | ablated and reweighted objectives, 34 runs (arms defined in `common.py`) |
| 4 | `job_matched.sh` | radius search to 2.27M templates and coverage, one task per trained flow (1-2 h each) |

**Aligned-spin 3D** (`al3d/`): `job_pairs.sh`, then `job_train.sh` (full objective and without `L_glob`, four
seeds each; index 3 is the paper's flow and is not retrained), then `job_score.sh`.

Then `python summary.py --live` summarises your runs.

## Notes

- All runs use the final-epoch checkpoint. The best-validation checkpoint (selected by the worst 10th percentile of
  the held-out distance ratio) inflated the latent distances in some runs and was a main source of variance.
- Placement stops when fewer than 1% of candidates are accepted (`stop_acceptance=0.01`); the release config's 0.10
  stops too early and leaves the low-mass corner empty.
- Trained flows, banks and fitting factors (about 4 GB) are not included.
