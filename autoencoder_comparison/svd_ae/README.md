# SVD + autoencoder (Cabass et al.), scored as the SVD baseline

The full pipeline of [Cabass et al.](https://github.com/giovanni-cabass/autoencoder_GWs) on the eccentric 5D
space: phase-SVD coefficients → distance-preserving autoencoder → lattice in the latent space. It is scored
exactly like Alexandra's plain SVD lattice (`svd_baseline/svd_grid_refined.py`) and drawn in one cumulative plot
with isobank and the parameter autoencoders of `../`.

| Curve | Map | Lattice | Fitting factor |
|---|---|---|---|
| isobank | flow on the parameters | greedy, r_target | true match, `scripts/evaluate_bank.py` |
| autoencoder on parameters | `../train_autoencoder.py` (Jacobian loss) | greedy, r_target | true match |
| SVD lattice, no autoencoder | none: the coefficients c0..c2 | BCC | phase overlap, as `svd_grid_refined.py` |
| SVD + autoencoder, latent 3 / 2 | Cabass's GW-example autoencoder, 10 coefficients → 3 or 2 | BCC / hexagonal | phase overlap, as `svd_grid_refined.py` |

The waveforms, draws, seeds, regressor and overlap are imported from or copied out of `svd_baseline/`, so the SVD
rows should reproduce `svd_baseline/results/svd_refined_ecc5d.json`. `lattice_table.md` prints her numbers next
to ours as a check.

All curves use the same 4,000 injections: the first 4,000 of seed 4002. The phase overlap has no extra time/phase
maximisation, so it is slightly pessimistic next to the true match.

Bank sizes are lattice cells reached by 2e8 draws, as `svd_grid_refined.py`. Each curve is drawn at the spacing
whose bank size is closest to isobank's.

**The autoencoder** follows `autoencoder_use_example/train_autoencoders_b_1_sb_0_to_2.py`:
- MLP, GELU, 64 hidden units, 4 layers;
- coefficients divided by std(c0), half held out (seed 1648);
- loss = reconstruction MSE + MSE of exp(−d)-weighted pairwise distances (a loss over all pairs, not the Jacobian
  loss of `../`);
- Adam 1e-3, 50,000 steps, keeps the best validation model.

It differs from theirs in three ways:
- pairs are taken within a random batch of 2048 per step;
- it runs in float64;
- it uses one bank instead of their sub-banks.

Templates are placed on a lattice in the latent space and decoded back to coefficients. The `no lattice` rows
show the autoencoder's own reconstruction error.

## Running at Nikhef

Needs `runs/eccentric_5d/ff_isobank.npy` and `bank_isobank.npy` from `../nikhef/comparison.dag`. The parameter
autoencoders' fitting factors are added if they exist.

```bash
cd /data/gravwav/lopezm/Projects/isobank && git pull
cd autoencoder_comparison/svd_ae/nikhef && condor_submit_dag svd_ae.dag
```

| Step | What | CPUs | Category |
|---|---|---|---|
| `data` | 30k waveforms, SVD, regressor (as `svd_grid_refined.py`) | 32 | medium |
| `train_ae3`, `train_ae2` | the autoencoder, latent 3 and 2 | 8 each | medium |
| `count` | 50 jobs, 4e6 draws each: cells of every lattice | 1 each | short |
| `evaluate` | fitting factors, `lattice_table.md`, `ff_cdf_svd.pdf`, `ff_summary_svd.md` | 1 | short |

Outputs go to `autoencoder_comparison/runs/eccentric_5d/svd_ae/`; logs to `logs/svd_ae_*`. The DAG uses at most
50 CPUs at once.
