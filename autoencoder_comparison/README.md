# isobank vs metric-preserving autoencoders (eccentric 5D)

The autoencoder of [Cabass et al.](https://github.com/giovanni-cabass/autoencoder_GWs) (Appendix B: MLP encoder
and decoder, reconstruction + Jacobian metric loss) trained on the isobank eccentric 5D space, with the same
data, metric, monitoring pairs, placement and fitting-factor protocol as isobank.

| File | What |
|---|---|
| `train_autoencoder.py` | trains the autoencoder; the metric loss on the decoder (`--side decoder`, Appendix B: J_Dᵀ M J_D = I) or on the encoder (`--side encoder`: J_Eᵀ J_E = M) |
| `make_bank_autoencoder.py` | greedy placement of `scripts/generate_bank.py`, with the encoder in place of the flow |
| `plot_comparison.py` | `training.pdf` (loss and d_z/d_proper during training) and `ff_cdf.pdf` (fitting factors) |
| `train_flow_ablation.py` | isobank's flow with its loss swapped (not part of the condor run; untested) |
| `nikhef/` | HTCondor DAG that runs everything at Nikhef |

## First results (laptop, 3000 epochs)

d_z/d_proper on the held-out monitoring pairs (isobank's flow: p10 = 0.96 in `configs/eccentric_5d.py`):

| Autoencoder | p10 | median | p90 |
|---|---|---|---|
| latent 5, metric loss on the encoder | 0.83 | 0.99 | 1.03 |
| latent 5, metric loss on the decoder (Appendix B) | 0.58 | 0.97 | 2.95 |
| latent 3, metric loss on the decoder | 0.04 | 0.18 | 1.01 |

The metric is very anisotropic (scaled eigenvalues up to 1e7, condition number ~1e11), which the decoder loss
handles worst. A 3D latent collapses distances: the space doesn't reduce to 3 dimensions at this spacing.

## Running at Nikhef

On an interactive node (`ssh stbc-i1`), with the code on `/project` (or `/data`):

```bash
# 1. code: the iclr branch with this folder
git clone -b iclr git@github.com:Alexandra-Wernersson/isobank.git /project/<group>/<user>/isobank
cd /project/<group>/<user>/isobank && mkdir -p logs

# 2. env (miniforge, if you don't have conda: https://github.com/conda-forge/miniforge)
mamba env create -f autoencoder_comparison/nikhef/environment.yml -p /project/<group>/<user>/conda/envs/isobank

# 3. edit the three lines at the top of autoencoder_comparison/nikhef/step.sub:
#    repo, env (the two paths above), group (your accounting group, from `id`)

# 4. submit
cd autoencoder_comparison/nikhef && condor_submit_dag comparison.dag
```

Monitor with `condor_q`, `condor_q -dag`, `tail -f comparison.dag.dagman.out`; each step logs to
`logs/<step>.out` and `logs/<step>.err`. A failed DAG writes `comparison.dag.rescue001`: fix the cause and
`condor_submit_dag comparison.dag` again to rerun only what did not finish.

The DAG (one CPU per step, `OMP_NUM_THREADS=1`):

| Step | What | Category |
|---|---|---|
| `train_isobank` | `scripts/train_flow.py` | long |
| `ae_data`, `train_ae_encoder`, `train_ae_decoder` | data, then the two autoencoders | short, medium |
| `bank_isobank`, `bank_ae_*` | the same greedy placement: radius r_target (config `MU`), stop at `STOP_ACCEPTANCE` | long |
| `ff_isobank`, `ff_ae_*` | `scripts/evaluate_bank.py`: 10,000 injections, seed 4002, k = 500 | medium |
| `plots` | `training.pdf` and `ff_cdf.pdf` in `autoencoder_comparison/runs/eccentric_5d/` | express |

The banks use the same placement and radius, so their sizes differ; the figure gives each bank's size. To add
mbank's fitting factors to the figure, rerun `plot_comparison.py` with `--extra-ff path:"mbank (2.95M)"`.
