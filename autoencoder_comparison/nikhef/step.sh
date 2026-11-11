#!/bin/bash
# One step of the isobank vs autoencoder comparison on the eccentric 5D space.
# Usage: step.sh STEP REPO ENV RADIUS_FACTOR   (run by comparison.dag, see ../README.md)
set -euo pipefail
STEP=$1 REPO=$2 ENV=$3 RADIUS=$4

export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$REPO"

CFG=configs/eccentric_5d.py
ISO=runs/eccentric_5d                         # isobank's own outputs (scripts/*.py)
AE=autoencoder_comparison/runs/eccentric_5d   # autoencoder outputs
mkdir -p "$ISO" "$AE"

case $STEP in
    train_isobank)
        python scripts/train_flow.py --config $CFG ;;
    ae_data)
        python autoencoder_comparison/train_autoencoder.py --config $CFG --prepare-only ;;
    train_ae_encoder|train_ae_decoder)
        python autoencoder_comparison/train_autoencoder.py --config $CFG --latent-dim 5 \
            --side "${STEP#train_ae_}" --epochs 3000 --monitor-every 50 ;;
    bank_isobank)
        python scripts/generate_bank.py --config $CFG --radius-factor "$RADIUS" \
            --out $ISO/bank_isobank.npy ;;
    bank_ae_encoder|bank_ae_decoder)
        side=${STEP#bank_ae_}
        python autoencoder_comparison/make_bank_autoencoder.py --config $CFG \
            --model $AE/autoencoder_latent5_$side.pt --radius-factor "$RADIUS" \
            --out $AE/bank_latent5_$side.npy ;;
    ff_isobank)
        python scripts/evaluate_bank.py --config $CFG --bank $ISO/bank_isobank.npy \
            --save-ff $ISO/ff_isobank.npy ;;
    ff_ae_encoder|ff_ae_decoder)
        side=${STEP#ff_ae_}
        python scripts/evaluate_bank.py --config $CFG --bank $AE/bank_latent5_$side.npy \
            --save-ff $AE/ff_latent5_$side.npy ;;
    plots)
        python autoencoder_comparison/plot_comparison.py --runs $AE \
            --isobank-log logs/train_isobank.out --isobank-ff $ISO/ff_isobank.npy \
            --isobank-bank $ISO/bank_isobank.npy ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
