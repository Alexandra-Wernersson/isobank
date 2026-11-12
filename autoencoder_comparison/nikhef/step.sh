#!/bin/bash
# One step of the isobank vs autoencoder comparison on the eccentric 5D space.
# Usage: step.sh STEP REPO ENV RADIUS_FACTOR [BANK PART N_PARTS]   (see ../README.md)
set -euo pipefail
STEP=$1 REPO=$2 ENV=$3 RADIUS=$4 BANK=${5:-} PART=${6:-} N_PARTS=${7:-}

export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$REPO"

CFG=configs/eccentric_5d.py
ISO=runs/eccentric_5d                         # isobank's own outputs (scripts/*.py)
AE=autoencoder_comparison/runs/eccentric_5d   # autoencoder outputs
mkdir -p "$ISO" "$AE"

bank_file() {   # isobank, ae_encoder or ae_decoder -> its bank and fitting factors
    case $1 in
        isobank) echo "$ISO/bank_isobank.npy $ISO/ff_isobank.npy" ;;
        ae_encoder|ae_decoder) echo "$AE/bank_latent5_${1#ae_}.npy $AE/ff_latent5_${1#ae_}.npy" ;;
        *) echo "unknown bank: $1" >&2; exit 1 ;;
    esac
}

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
    ff_part)   # one of N_PARTS slices of the injections (ff_parts.sub)
        read -r bank _ <<< "$(bank_file "$BANK")"
        python autoencoder_comparison/ff_parallel.py part --config $CFG --bank "$bank" \
            --parts "$N_PARTS" --part "$PART" --out-dir "$AE/ff_parts_$BANK" ;;
    ff_join_isobank|ff_join_ae_encoder|ff_join_ae_decoder)
        which=${STEP#ff_join_}
        read -r _ ff <<< "$(bank_file "$which")"
        python autoencoder_comparison/ff_parallel.py join --out-dir "$AE/ff_parts_$which" --save-ff "$ff" ;;
    plots)
        python autoencoder_comparison/plot_comparison.py --runs $AE \
            --isobank-log logs/train_isobank.out --isobank-ff $ISO/ff_isobank.npy \
            --isobank-bank $ISO/bank_isobank.npy ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
