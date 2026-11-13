#!/bin/bash
# The fitting factors of one variant's bank in parallel parts (../../autoencoder_comparison/ff_parallel.py):
# the same injections and results as `step.sh ff`, which does them in one job.
# Usage: ff_step.sh part|join NAME PART N_PARTS ISOBANK ENV WORK
set -euo pipefail
MODE=$1 NAME=$2 PART=$3 N_PARTS=$4 ISOBANK=$5 ENV=$6 WORK=$7

export PATH=$ENV/bin:$PATH
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
cd "$ISOBANK"

FF="python autoencoder_comparison/ff_parallel.py"
RUN=$WORK/$NAME
case $MODE in
    part) $FF part --config configs/eccentric_5d.py --bank "$RUN/bank.npy" --parts "$N_PARTS" \
              --part "$PART" --out-dir "$RUN/ff_parts" ;;
    join) $FF join --out-dir "$RUN/ff_parts" --parts "$N_PARTS" --save-ff "$RUN/ff.npy" ;;
    *)    echo "unknown mode: $MODE" >&2; exit 1 ;;
esac
