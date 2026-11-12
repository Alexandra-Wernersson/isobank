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

**Sub-banks** (`--subbanks K`): as Cabass's 3 sub-banks, the waveforms are split into K slices of equal count in
c0, each with its own autoencoder, centred on its mean and divided by its own std(c0). Each network then only has
to be precise over a narrow range of c0. A draw or an injection goes to the sub-bank of its c0, and the bank size
is the sum over sub-banks.

It differs from theirs in four ways:
- their sub-banks each have their own SVD basis; ours share Alexandra's single basis, sliced and centred;
- with one bank (13,000 training waveforms), pairs are taken within a random batch of 2048 per step (each
  sub-bank trains on all its waveforms, as theirs);
- it runs in float64;
- the best model is picked every 100 steps on at most 2048 held-out waveforms (theirs: every epoch, all of them).

Templates are placed on a lattice in the latent space and decoded back to coefficients. The `no lattice` rows
show the autoencoder's own reconstruction error.

## Results (Nikhef, 12 November 2026)

`svd_ae.dag`, one bank (`../runs/eccentric_5d/svd_ae/lattice_table.md`, `ff_summary_svd.md`):

| Bank | Templates | Coverage, FF ≥ 0.97 | 1st pct |
|---|---|---|---|
| isobank (Nikhef retrain) | 3.01M | 98.38 ± 0.20 % | 0.963 |
| SVD lattice, no autoencoder | 3.33M | 99.70 ± 0.09 % | 0.974 |
| SVD + autoencoder, latent 3, one bank | 10k | 1.27 ± 0.18 % | < 0 |
| autoencoder on parameters, encoder loss | 42k | 17.70 ± 0.60 % | 0.800 |
| autoencoder on parameters, decoder loss | 31.8M | 98.83 ± 0.17 % | 0.968 |

- The SVD rows reproduce `svd_baseline/results/svd_refined_ecc5d.json` (same coverage, sizes within 0.02 %).
- With one bank the autoencoder does not train precisely enough: its validation loss stops at 7.3e-6 after 5,000
  steps, an error of 1–2 per coefficient (a template needs about 0.1), so even without a lattice its fitting
  factors are near 0. This is why `svd_ae_subbanks.dag` adds Cabass's sub-banks.
- Locally (300 waveforms per sub-bank, 3 sub-banks), sub-banks cut the error on c0 and c1 by 3–5×; c2 is then
  the limit.

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

Then the sub-banks, 10 and then 20, in one DAG that reuses the `data` step and the one-bank models:

```bash
cd /data/gravwav/lopezm/Projects/isobank && git pull
cd autoencoder_comparison/svd_ae/nikhef && condor_submit_dag svd_ae_subbanks.dag
```

| Step | What | CPUs | Category |
|---|---|---|---|
| `train_ae3_sb0` … `train_ae2_sb9` | 10 sub-banks × latent 3 and 2 | 2 each | medium |
| `count10`, `evaluate10` | bank sizes and plot with the models so far (one bank, 10 sub-banks): `lattice_table_sb10.md`, `ff_summary_svd_sb10.md`, `ff_cdf_svd_sb10.pdf` | 1 each | short |
| `train_ae3_sb0of20` … `train_ae2_sb19of20` | 20 sub-banks × latent 3 and 2 | 1 each | medium |
| `count20`, `evaluate20` | the same with every model (one bank, 10 and 20 sub-banks): files `..._sb20` | 1 each | short |

At most 40 CPUs at once (50 while counting); about 6–8 h in all. `count` and `evaluate` use every complete set of
models (`--models auto` in `step.sh`). The plot shows the latent-3 sub-bank models; `lattice_table.md` lists all
of them.
