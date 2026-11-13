# isobank's run-to-run variance (eccentric 5D)

Reviewer 6RHG: "Injection error bars are reported; flow retraining variance is not." mbank has three repeats with
the paper's settings (`mbank_comparison/tuning/`, 93.5–94.8 %); this gives isobank the same.

| Run | Flow | Placement seed | What it measures |
|---|---|---|---|
| reference (run 0) | `runs/eccentric_5d/` (comparison DAG) | 0 | — |
| `seed1`, `seed2`, `seed3` | retrained with `scripts/train_flow.py --seed s` | 0 | retraining: network initialisation and batch order |
| `place1`, `place2` | the reference flow | 1, 2 | placement alone |

Everything else is the reference's:
- the same r_target (copied, not recalibrated);
- the same training data and pairs, whose seeds are fixed in `train_flow.py` and the config;
- radius = r_target;
- the same 10,000 injections of seed 4002, scored as `scripts/evaluate_bank.py` (`ff_parallel.py`, 10 parts).

`summarise.py` writes `runs/eccentric_5d_retrain/summary.md`: per run, bank size, coverage, 1st percentile and
worst FF, and the flow's best held-out p10; then the mean ± std over the trainings and over the placements.

Two options make this possible without touching the reference run:
- `ISOBANK_OUT_DIR` (in `configs/eccentric_5d.py`) sends a run's checkpoints and banks to its own folder;
- `--seed` (in `scripts/train_flow.py`) seeds the network.

Without them, the scripts behave as before.

## Running at Nikhef

Needs `runs/eccentric_5d/flow_best.pt`, `r_target.npy`, `bank_isobank.npy` and `ff_isobank.npy` from
`autoencoder_comparison/nikhef/comparison.dag` (done).

```bash
cd /data/gravwav/lopezm/Projects/isobank && git pull
cd retrain_variance/nikhef && condor_submit_dag retrain.dag
```

| Step | Jobs | CPUs | Category |
|---|---|---|---|
| `train_seed1` … `train_seed3` | 3 | 1 each (as the reference) | long |
| `bank_*` (3 after their training, 2 at once) | 5 | 1 each | long |
| `ffparts_*`, `ffjoin_*` | 5 × 10 + 5 | 1 each | medium, express |
| `summary` | 1 | 1 | express |

At most 50 CPUs at once. Outputs: `runs/eccentric_5d_retrain/<run>/` (`flow_best.pt`, `bank.npy`, `ff.npy`) and
`summary.md`; logs: `logs/retrain_<step>_<run>_<i>.out`.
