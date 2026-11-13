# mbank tuning sweep (eccentric 5D)

How much does mbank improve with tuning? Each variant retrains mbank's flow with one change to the loss or
the training, then places a bank with the paper's placement (random, mm 0.97, covering fraction 0.9) and scores
it with isobank's `scripts/evaluate_bank.py`: 10,000 injections, seed 4002, k = 500. This is the same plot as
`figures/bank_size_vs_coverage.png`, but varying the training instead of the placement.

| Variant | Change from the paper's `base.ini` (= `../mbank_placement/ecc5D_mbank_native.ini`) |
|---|---|
| `paper_a`, `paper_b`, `paper_c` | none: three repeats, the run-to-run scatter |
| `weighted_0.5`, `weighted_1` | loss `weighted_ll_mse`, weights (ℓ − min ℓ)^0.5 (mbank's) or ^1 |
| `quantile_0.5`, `quantile_0.99` | starting value of the constant C: that quantile of the log volumes (mbank: 0.9) |
| `padding_0.001`, `padding_0.05` | margin of the flow's range around the box, per side (mbank: 0.01) |
| `lr_0.0005`, `batch_1000` | learning rate (paper: 0.005), batch size (paper: 5000) |
| `data_50k` | 50,000 training points (paper: 15,000) |
| `is_wmax_10`, `is_no_clip` | `paper_a`'s flow, covering fraction by importance sampling, weights clipped at W_max = 10 or not at all |

The variants are the rows of `variants.tsv`. To change them, edit it and run `python make_dag.py`, which
rewrites `tuning.dag`.

Outputs, in `WORK`:
- `tuning.pdf`: bank size against coverage, with isobank (from `runs/eccentric_5d/` if the comparison DAG has
  made its bank, else the paper's) and the paper's mbank bank;
- `tuning_loss.pdf`: validation loss of each flow;
- `tuning.csv`: per variant, size, coverage and the flow diagnostics of `diagnose_flow.py` (fit error,
  fraction of samples inside the box, `C − C*`, mbank's volume and the corrected one).

## Results (Nikhef, 13 November 2026)

`results/tuning.csv`, `results/tuning.pdf`, `results/tuning_loss.pdf`; the reviewer-facing figure is
`results/tuning_summary.pdf` (`plot_tuning_summary.py`).

| Variant | Templates | Coverage, FF ≥ 0.97 |
|---|---|---|
| `paper_a`, `paper_b`, `paper_c` | 3.09M, 2.84M, 3.01M | 94.78, 93.51, 93.73 % |
| `weighted_0.5`, `weighted_1` | 2.99M, 3.03M | 94.46, 94.37 % |
| `quantile_0.5`, `quantile_0.99` | 2.92M, 3.01M | 94.00, 93.49 % |
| `padding_0.001`, `padding_0.05` | 3.05M, 3.07M | 93.80, 94.30 % |
| `lr_0.0005`, `batch_1000` | 2.84M, 2.98M | 93.41, 93.61 % |
| `data_50k` | 3.04M | 94.29 % |
| `is_wmax_10`, `is_no_clip` | 2.99M, 3.11M | 94.20, 94.32 % |
| isobank (Nikhef retrain) | 3.01M | 98.4 % |

Errors are about ±0.23 %. Every variant is within the scatter of the three runs with the paper's settings (93.5–94.8 %):
no change to the loss, training or placement brings mbank near isobank.

- The weighted loss gives the best flow fit (median |r| 0.019, against 0.03–0.05 for the paper's runs) and a
  coverage at the top of the scatter, but one run per variant cannot separate it from the repeats.
- The fraction of flow samples inside the box (`samples_inside`, about 4 %) does not depend on the padding; it
  follows the starting quantile of C (19 % at 0.5, 2 % at 0.99). The volume inside the box is the same in every
  variant (log10 ≈ −0.68), so this changes mbank's reported volume and wasted proposals, not the bank.

## mbank

The variants need options added to mbank in the fork [MelissaLP/mbank](https://github.com/MelissaLP/mbank):
`--constant-quantile`, `--boundary-padding`, `--weight-exponent`, `--importance-sampling` and `--w-max`.
Their defaults are mbank's hard-coded values, so the paper's runs are unchanged.

The paper's mbank is upstream commit 9df83eb plus `../../mbank_local_changes.patch` (`../../environment.yml`).
The fork has the patch applied. Of its changes, two affect results: gradient clipping (max norm 5) in the
flow training, and points outside the flow's range no longer raise an error.

## Running at Nikhef

On `stbc-i1`:

```bash
# code: this repository (iclr) and the mbank fork
git clone -b iclr git@github.com:Alexandra-Wernersson/isobank.git /data/gravwav/lopezm/Projects/isobank
git clone git@github.com:MelissaLP/mbank.git /data/gravwav/lopezm/Projects/mbank

# env isobank-mbank: the pins of the isobank env plus mbank's dependencies, and mbank from the fork
# (a separate env, so the isobank env of the autoencoder runs is left as it is)
E=/data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank-mbank
conda env create -f /data/gravwav/lopezm/Projects/isobank/environment.yml -p $E
$E/bin/pip install --no-deps -e /data/gravwav/lopezm/Projects/mbank
$E/bin/python -c "import mbank, torch, lalsimulation, glasflow; print(mbank.__file__)"   # a path in Projects/mbank

# check the paths at the top of mbank_comparison/tuning/step.sub, then
mkdir -p /data/gravwav/lopezm/Projects/mbank_tuning/logs                 # the `work` folder of step.sub
cd /data/gravwav/lopezm/Projects/isobank/mbank_comparison/tuning && condor_submit_dag tuning.dag
```

The fitting factors use isobank's flow to shortlist templates. The `isobank_flow` step trains it, unless
`runs/eccentric_5d/flow_best.pt` already exists. If the comparison DAG (`autoencoder_comparison/`) is also
running, submit this one once its `train_isobank` step has finished, so the two don't train it at once.

The steps run mbank's scripts from the fork's `bin/`, through `../mbank_placement/run_mbank_cli.py` as the
paper's runs did, one CPU each with `OMP_NUM_THREADS=1`. Logs: `WORK/logs/<step>_<variant>.out` and `.err`.

The fitting factors of each variant run in 3 parallel jobs (`FF_PARTS` in `make_dag.py`; `ff.sub`,
`ff_step.sh`, `../../autoencoder_comparison/ff_parallel.py`), then a join that writes `ff.npy`. Results are
identical to one `evaluate_bank.py` run, but take about 1.5 h instead of 4 h. The sweep uses at most 42 CPUs at
once. Each job asks for 8 GB, the placements for 32 GB (`MEMORY` in `make_dag.py`). Logs: `WORK/logs/ff_part_<variant>_<i>.out`, `ff_join_<variant>_0.out`.

A command-line value equal to mbank's parser default is overridden by `base.ini`: e.g. `--learning-rate 0.001`
(mbank's default) would leave the ini's 0.005 in place. Check `args_*.json` in a variant's folder for the
values a run used.
