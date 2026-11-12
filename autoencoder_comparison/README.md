# isobank vs metric-preserving autoencoders (eccentric 5D)

The autoencoder of [Cabass et al.](https://github.com/giovanni-cabass/autoencoder_GWs) (Appendix B: MLP encoder
and decoder, reconstruction + Jacobian metric loss) trained on the isobank eccentric 5D space, with the same
data, metric, monitoring pairs, placement and fitting-factor protocol as isobank.

| File | What |
|---|---|
| `train_autoencoder.py` | trains the autoencoder; the metric loss on the decoder (`--side decoder`, Appendix B: J_Dᵀ M J_D = I) or on the encoder (`--side encoder`: J_Eᵀ J_E = M) |
| `make_bank_autoencoder.py` | greedy placement of `scripts/generate_bank.py`, with the encoder in place of the flow |
| `plot_comparison.py` | `training.pdf` (loss and d_z/d_proper during training) and `ff_cdf.pdf` (fitting factors) |
| `latent_neighbours.py` | folding test: true match of the points a latent space puts within r_target |
| `ff_parallel.py` | `scripts/evaluate_bank.py` split over parallel jobs, with identical results (`nikhef/ff_parts.sub`, `nikhef/ff_decoder.dag`) |
| `summarise_ff.py` | spread of the fitting factors: mean ± std, median, 1st/5th/10th percentiles, coverage |
| `train_flow_ablation.py` | isobank's flow with its loss swapped (not part of the condor run; untested) |
| `nikhef/` | HTCondor DAG that runs everything at Nikhef |
| `svd_ae/` | Cabass's full pipeline (SVD coefficients → autoencoder → lattice), scored as `svd_baseline/`; one cumulative plot with isobank, the SVD lattice and the parameter autoencoders |

## Status (12 November 2026)

**Nikhef run.** One job went over 16 GB and was held, then released without changing its memory; it then
finished. It seems to be `ff_ae_encoder`, but on the laptop that step peaks at 0.5 GB (matches in chunks of
20,000, a 41,796-template bank), so the cause is unclear: see command 3 below. Results so far:

| Bank | Templates | Coverage, FF ≥ 0.97 | FF mean ± std | 1st / 5th pct | Notes |
|---|---|---|---|---|---|
| isobank | ? | ? | ? | ? | `ff_isobank` finished, not read yet |
| autoencoder, encoder loss | 41,796 | 17.75 ± 0.38 % | ? | ? | worst FF 0.557; miss rate flat from k = 20; shortlist reach ≥ 24.7 r_target |
| autoencoder, decoder loss | ? | ? | ? | ? | `ff_ae_decoder` finished, not read yet |

The worst FF is the minimum of one set of 10,000 injections and changes a lot from set to set; the mean ± std
and the low percentiles (from `summarise_ff.py`) describe the spread more stably.

**Folding test (laptop models, not the Nikhef ones).** Random points that the latent space puts within r_target,
and their true waveform match. A faithful map gives near-identical waveforms:

| Latent space | Pairs within r_target | Median match | Match < 0.97 | Match < 0.5 |
|---|---|---|---|---|
| autoencoder, encoder loss | 4,166,204 | 0.676 | 97.8 % | 8.3 % |
| autoencoder, decoder loss | 225 | 0.839 | 75.1 % | 26.2 % |

Both autoencoders fold: they get local distances right (table below) but put distant, badly matching waveforms
next to each other, so placement leaves those regions uncovered. That explains the encoder bank: 55x smaller than
isobank's, 18 % coverage. The bad pairs are far apart in the parameters (median 70 % of the s1z range). The
decoder latent space is also stretched, so its bank should be large.

`notes/autoencoder_explained.pdf` explains this in simple terms (not committed: `*.pdf` is ignored).

**Decoder fitting factors in parallel.** `ff_ae_decoder` does its 5 million matches on one CPU, about 4 hours.
`nikhef/ff_decoder.dag` does the same in 20 jobs (each about 15 minutes of matches), then the plots. To switch,
remove only that job, which ends `comparison.dag` (all its other steps are done) and touches nothing else.
Remove it before `git pull`: the pull changes `nikhef/step.sh`, which the running job is still reading.

```bash
cd /data/gravwav/lopezm/Projects/isobank
condor_q -constraint 'DAGNodeName == "ff_ae_decoder"' -af ClusterId DAGNodeName   # check: one job
condor_rm -constraint 'DAGNodeName == "ff_ae_decoder"'
git pull
cd autoencoder_comparison/nikhef && condor_submit_dag ff_decoder.dag
```

**Next**, at Nikhef, so that every number comes from the Nikhef runs:

```bash
cd /data/gravwav/lopezm/Projects/isobank && git pull

# 1. bank sizes, and the spread of the fitting factors of the three banks
grep -E "^bank" logs/ff_isobank.out logs/ff_ae_encoder.out logs/ff_ae_decoder.out
A=autoencoder_comparison/runs/eccentric_5d
/data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank/bin/python autoencoder_comparison/summarise_ff.py \
    runs/eccentric_5d/ff_isobank.npy:isobank $A/ff_latent5_encoder.npy:"autoencoder, encoder loss" \
    $A/ff_latent5_decoder.npy:"autoencoder, decoder loss"

# 2. folding test on the Nikhef models, with isobank's flow as the reference
OMP_NUM_THREADS=1 /data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank/bin/python autoencoder_comparison/latent_neighbours.py \
    --config configs/eccentric_5d.py --isobank runs/eccentric_5d/flow_best.pt \
    --autoencoder $A/autoencoder_latent5_encoder.pt --autoencoder $A/autoencoder_latent5_decoder.pt

# 3. the held job: what it exceeded and its peak memory (MB)
condor_history -constraint 'DAGNodeName == "ff_ae_encoder"' \
    -af ClusterId RequestMemory MemoryUsage ResidentSetSize_RAW NumJobStarts HoldReason
grep -A3 "Job was held" logs/condor.log
```

Then: fill in the tables above with these numbers, rebuild the PDF from them, and check `training.pdf` and
`ff_cdf.pdf` once the `plots` step has run.

## Local distances (laptop, 3000 epochs)

A quick check that the latent space preserves distances, before placing any bank. It uses ~3000 held-out pairs at
placement scale (0.5–1.5 r_target), built as isobank's monitoring pairs. For each pair,

    ratio = latent distance / true proper distance

which is 1 for a perfect map. The table gives its percentiles:

- **median**: the typical pair; near 1 means distances are right on average;
- **p10**: the most shrunk pairs; below 1 means templates too sparse there, so missed signals
  (isobank's flow: p10 = 0.96, in `configs/eccentric_5d.py`);
- **p90**: the most stretched pairs; above 1 means wasted templates.

| Autoencoder | p10 | median | p90 | In words |
|---|---|---|---|---|
| latent 5, metric loss on the encoder | 0.83 | 0.99 | 1.03 | right on average; the worst 10 % shrunk by ≥ 17 % |
| latent 5, metric loss on the decoder (Appendix B) | 0.58 | 0.97 | 2.95 | right on average; large errors both ways |
| latent 3, metric loss on the decoder | 0.04 | 0.18 | 1.01 | fails: distances collapse |

"Encoder" or "decoder" is the network the metric loss is applied to (the Appendix B example uses the decoder);
"latent 3" compresses the 5D space, as the 10 → 2 GW example of the paper does.

The metric is very anisotropic (scaled eigenvalues up to 1e7, condition number ~1e11), which the decoder loss
handles worst. A 3D latent collapses distances: the space doesn't reduce to 3 dimensions at this spacing.

This doesn't say whether the errors cost coverage: the condor run does, by placing a bank with each method and
computing fitting factors. It also retrains isobank and scores it on the same pairs.

## Running at Nikhef

On an interactive node (`ssh stbc-i1`):

```bash
# 1. code: the iclr branch with this folder
git clone -b iclr git@github.com:Alexandra-Wernersson/isobank.git /data/gravwav/lopezm/Projects/isobank
cd /data/gravwav/lopezm/Projects/isobank && mkdir -p logs

# 2. env (miniforge, if you don't have conda: https://github.com/conda-forge/miniforge)
conda env create -f autoencoder_comparison/nikhef/environment.yml -p /data/gravwav/lopezm/cmb_env/miniforge3/envs/isobank

# 3. check the two paths at the top of autoencoder_comparison/nikhef/step.sub (repo, conda_env)

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
| `plots` | `training.pdf`, `ff_cdf.pdf` and `ff_summary.md` in `autoencoder_comparison/runs/eccentric_5d/` | express |

The banks use the same placement and radius, so their sizes differ; the figure gives each bank's size. To add
mbank's fitting factors to the figure, rerun `plot_comparison.py` with `--extra-ff path:"mbank (2.95M)"`.
