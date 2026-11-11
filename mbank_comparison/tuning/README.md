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

## mbank

The variants need options added to mbank in the fork [MelissaLP/mbank](https://github.com/MelissaLP/mbank):
`--constant-quantile`, `--boundary-padding`, `--weight-exponent`, `--importance-sampling` and `--w-max`.
Their defaults are mbank's hard-coded values, so the paper's runs are unchanged.

`../../environment.yml` says that the paper's mbank is upstream commit 9df83eb plus `mbank_local_changes.patch`.
The fork is at 9df83eb, but the patch isn't in the repository yet: apply it to the fork when you have it.

## Running at Nikhef

On `stbc-i1`:

```bash
# code: this repository (iclr) and the mbank fork
git clone -b iclr git@github.com:Alexandra-Wernersson/isobank.git /data/gravwav/lopezm/Projects/isobank
git clone git@github.com:MelissaLP/mbank.git /data/gravwav/lopezm/Projects/mbank

# env: the versions of the Snellius runs, plus mbank from the fork
mamba env create -f /data/gravwav/lopezm/Projects/isobank/environment.yml -p /data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank-mbank
/data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank-mbank/bin/pip install --no-deps -e /data/gravwav/lopezm/Projects/mbank

# check the paths at the top of mbank_comparison/tuning/step.sub, then
mkdir -p /data/gravwav/lopezm/Projects/mbank_tuning/logs                 # the `work` folder of step.sub
cd /data/gravwav/lopezm/Projects/isobank/mbank_comparison/tuning && condor_submit_dag tuning.dag
```

The fitting factors use isobank's flow to shortlist templates. The `isobank_flow` step trains it, unless
`runs/eccentric_5d/flow_best.pt` already exists. If the comparison DAG (`autoencoder_comparison/`) is also
running, submit this one once its `train_isobank` step has finished, so the two don't train it at once.

The steps run mbank's scripts from the fork's `bin/`, through `../mbank_placement/run_mbank_cli.py` as the
paper's runs did, one CPU each with `OMP_NUM_THREADS=1`. Logs: `WORK/logs/<step>_<variant>.out` and `.err`.

A command-line value equal to mbank's parser default is overridden by `base.ini`: e.g. `--learning-rate 0.001`
(mbank's default) would leave the ini's 0.005 in place. Check `args_*.json` in a variant's folder for the
values a run used.
