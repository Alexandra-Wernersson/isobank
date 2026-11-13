#!/bin/bash
# One step of retrain.dag: isobank retrained and re-placed with other seeds (see ../README.md).
# Usage: step.sh STEP RUN REPO ENV [PART N_PARTS]
#   RUN = seed<s>  (flow retrained with --seed s, bank placed with seed 0, as runs/eccentric_5d)
#       | place<p> (the flow of runs/eccentric_5d, bank placed with --seed p)
set -euo pipefail
STEP=$1 RUN=$2 REPO=$3 ENV=$4 PART=${5:-} N_PARTS=${6:-}

export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$REPO"

CFG=configs/eccentric_5d.py
REF=runs/eccentric_5d                         # the comparison run: flow, r_target, bank (placement seed 0)
D=runs/eccentric_5d_retrain/$RUN              # this run's bank and fitting factors
[[ $RUN == - ]] || mkdir -p "$D"
case $RUN in
    seed*)  export ISOBANK_OUT_DIR=$REPO/$D; SEED=${RUN#seed}; PLACE_SEED=0 ;;   # its own flow
    place*) export ISOBANK_OUT_DIR=$REPO/$REF; PLACE_SEED=${RUN#place} ;;          # the reference flow
    -)      ;;
    *)      echo "unknown run: $RUN" >&2; exit 1 ;;
esac

case $STEP in
    train)      # same r_target, data and pairs as the reference; only the network seed changes
        cp -n $REF/r_target.npy "$D/"
        python scripts/train_flow.py --config $CFG --seed "$SEED" ;;
    bank)
        python scripts/generate_bank.py --config $CFG --radius-factor 1.0 --seed "$PLACE_SEED" \
            --out "$D/bank.npy" ;;
    ff_part)
        python autoencoder_comparison/ff_parallel.py part --config $CFG --bank "$D/bank.npy" \
            --parts "$N_PARTS" --part "$PART" --out-dir "$D/ff_parts" ;;
    ff_join)
        python autoencoder_comparison/ff_parallel.py join --out-dir "$D/ff_parts" --parts "$N_PARTS" \
            --save-ff "$D/ff.npy" ;;
    summary)
        python retrain_variance/summarise.py --ref $REF --runs runs/eccentric_5d_retrain --logs logs ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
