# isobank

**Template banks from approximately isometric normalizing flows.**

Placing templates on a waveform manifold is a sphere covering problem in a
curved space: distances depend on position through the waveform metric, so
every acceptance test costs a metric evaluation. `isobank` trains a normalizing
flow so that **Euclidean distance in its latent space approximates the proper
distance of the waveform match metric**. Placement then becomes sphere packing
with Euclidean distance checks, and nearest-template lookups become k-d tree
queries — with no metric or waveform evaluated during placement.

For the five-dimensional eccentric BNS/NSBH case, the flow trains in about 2.5 hours on 32 CPU cores, and template placement then takes a few minutes.

## Install

```bash
git clone https://github.com/Alexandra-Wernersson/isobank.git
cd isobank
pip install -e .
```

`lalsuite` supplies the waveform models; everything else is standard scientific
Python. The pipeline is CPU-only. 

Conda users can instead run `conda env create -f environment.yml`.

## Quick start

Two cases ship with the package: `configs/aligned_3d.py` (3D aligned-spin BBH,
`IMRPhenomD`, design sensitivity) and `configs/eccentric_5d.py` (5D eccentric
BNS/NSBH, `TaylorF2Ecc`, O3 sensitivity). The full pipeline is three commands:

```bash
# 1. calibrate the placement radius and train the flow
#    (~2.5 h on 32 CPU cores for the 5D case; minutes for the 3D one)
python scripts/train_flow.py --config configs/eccentric_5d.py

# 2. place a bank (minutes)
python scripts/generate_bank.py --config configs/eccentric_5d.py \
    --radius-factor 1.45 --out runs/eccentric_5d/bank.npy

# 3. measure coverage against true waveform matches
python scripts/evaluate_bank.py --config configs/eccentric_5d.py \
    --bank runs/eccentric_5d/bank.npy --n-inj 10000 --k 500
```

Step 1 writes the calibrated radius and the flow checkpoints into
`runs/<case>/`; steps 2 and 3 read them from there. Placement is deterministic
given `--seed`, so a bank can be reproduced exactly.

## How it works

1. **Calibrate.** `r_target` is the proper distance at which the mean *true*
   waveform mismatch equals the target mismatch `mu`. The quadratic metric
   approximation is not accurate at placement separations, so this is measured,
   not derived.
2. **Train.** The flow is fitted to preserve distances at the scales that matter
   for placement: a scale invariant global term, plus metric-guided local,
   boundary and (optionally) edge pairs, trained with a relative loss that
   gives most weight to the worst fitting pairs.
3. **Place.** Candidates are drawn, mapped through the flow, and accepted when
   their nearest existing template is farther than the placement radius.
4. **Validate.** Coverage is always measured with true matches, never inferred
   from the flow.

### Bank size and coverage

`--radius-factor` sets the placement radius in units of `r_target`. A factor of
1 places at the calibrated radius, which over-covers: a typical signal then sits
well inside that radius (median 0.64 `r_target` for the 5D case), so it matches
far above threshold. Larger factors give smaller banks whose typical
signal-to-template distance sits near `r_target`. The factor trades bank size
against coverage, and coverage must always be measured.

### Reading the evaluation output

`evaluate_bank.py` computes exact matches against the `k` nearest templates in
latent space. Two printed diagnostics say whether the result can be trusted:

* **shortlist reach** — distance to the k-th candidate in units of `r_target`;
  must exceed the separation beyond which no pair can reach the threshold.
* **miss rate vs k** — must be flat at the chosen `k`; if it is still falling,
  raise `k`, as the quoted miss rate is otherwise an upper bound.

## Figures

```bash
python figures/plot_bank_map.py --config configs/eccentric_5d.py \
    --bank runs/eccentric_5d/bank.npy --out bank_map.pdf
python figures/plot_ff_cdf.py --ff runs/eccentric_5d/ff.npy:"isobank" --out ff_cdf.pdf
python figures/plot_match_vs_latent.py --config configs/eccentric_5d.py --out match.pdf
```

The last one also prints the mean match at `r_target`, which should equal the
minimal match if calibration and flow are behaving (a useful end-to-end check).

## Adding a parameter space

Copy a file in `configs/` and set the parameter ranges and waveform model, the
PSD, the flow size and coupling masks, the pair counts and the output directory.
Anything left undefined falls back to a default in the scripts. `PARAM_LABELS`
gives the axis labels used by the figure scripts.


## License and provenance

GPLv3 (see `LICENSE`). `isobank/metric/{metric,handlers,utils}.py` are
adapted from [mbank](https://github.com/stefanoschmidt1995/mbank) (GPLv3),
trimmed to the metric and parameter handling used here.
