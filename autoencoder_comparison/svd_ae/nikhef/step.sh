#!/bin/bash
# One step of svd_ae.dag (Cabass et al.'s SVD + autoencoder pipeline, ../svd_ae.py).
# Usage: step.sh STEP REPO ENV CPUS [PART N_PARTS]   (see ../README.md)
set -euo pipefail
STEP=$1 REPO=$2 ENV=$3 CPUS=$4 PART=${5:-} N_PARTS=${6:-}

export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export SLURM_CPUS_PER_TASK=$CPUS   # worker processes / torch threads, as read by svd_baseline/*.py
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$REPO"

RUN="python autoencoder_comparison/svd_ae/svd_ae.py"
OUT=autoencoder_comparison/runs/eccentric_5d/svd_ae
ISO=runs/eccentric_5d                         # isobank, from comparison.dag
AE=autoencoder_comparison/runs/eccentric_5d   # autoencoder on the parameters, from comparison.dag
mkdir -p "$OUT"
MODELS=auto      # count and evaluate: every autoencoder whose files are all there (svd_ae.py --models)

case $STEP in
    data)                 $RUN data --out $OUT ;;
    train_ae3|train_ae2)  $RUN train --out $OUT --latent-dim "${STEP#train_ae}" ;;
    train_ae*_sb*)        # train_ae<latent>_sb<j>, sub-bank j of 10, or train_ae<latent>_sb<j>of<K>
        k=${STEP#train_ae}; k=${k%%_sb*}; j=${STEP##*_sb}; n_sb=10
        if [[ $j == *of* ]]; then n_sb=${j#*of}; j=${j%of*}; fi
        $RUN train --out $OUT --latent-dim "$k" --subbanks "$n_sb" --subbank "$j" ;;
    count)                $RUN count --out $OUT --parts "$N_PARTS" --part "$PART" --models $MODELS ;;
    evaluate)
        extra=()
        for side in encoder decoder; do       # the parameter autoencoders, where comparison.dag has scored them
            ff=$AE/ff_latent5_$side.npy
            [[ -f $ff ]] || continue
            n=$(python -c "import numpy as np; n = len(np.load(\"$AE/bank_latent5_$side.npy\", mmap_mode=\"r\")); print(f'{n / 1e6:.2f}M' if n >= 1e6 else f'{n / 1e3:.0f}k')")
            extra+=(--extra-ff "$ff:autoencoder on parameters, $side loss ($n)")
        done
        $RUN evaluate --out $OUT --parts "$N_PARTS" --models $MODELS --isobank-ff $ISO/ff_isobank.npy \
            --isobank-bank $ISO/bank_isobank.npy "${extra[@]}" ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
