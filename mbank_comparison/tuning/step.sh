#!/bin/bash
# One step of the mbank tuning sweep (tuning.dag, see README.md).
# Usage: step.sh STEP NAME ISOBANK MBANK ENV WORK
set -euo pipefail
STEP=$1 NAME=$2 ISOBANK=$3 MBANK=$4 ENV=$5 WORK=$6

export PATH=$ENV/bin:$PATH
export PYTHONPATH=$MBANK${PYTHONPATH:+:$PYTHONPATH}
export OMP_NUM_THREADS=1      # lalsimulation 7.0.0 TaylorF2Ecc is not reproducible with more threads
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

HERE=$ISOBANK/mbank_comparison/tuning
INI=$HERE/base.ini
PSD=$ISOBANK/mbank_comparison/mbank_placement/aligo_O3actual_H1.txt
MBANK_CLI="python $ISOBANK/mbank_comparison/mbank_placement/run_mbank_cli.py"   # as the paper's runs
RUN=$WORK/$NAME

field() {   # column $1 of the variant's row in variants.tsv, "" for "-"
    local v
    v=$(awk -F'\t' -v n="$NAME" -v c="$1" '$1 == n {print $c}' "$HERE/variants.tsv")
    [ "$v" = "-" ] && v=""
    echo "$v"
}

case $STEP in
    isobank_flow)   # isobank's flow, used to shortlist templates in evaluate_bank.py
        cd "$ISOBANK"
        if [ -f runs/eccentric_5d/flow_best.pt ]; then
            echo "using the existing runs/eccentric_5d/flow_best.pt"
        else
            python scripts/train_flow.py --config configs/eccentric_5d.py
        fi ;;
    dataset)
        n=${NAME#d}; n=${n%k}000
        mkdir -p "$WORK/datasets" && cd "$WORK/datasets"
        $MBANK_CLI "$MBANK/bin/mbank_generate_flow_dataset" "$INI" --run-dir "$WORK/datasets/" \
            --dataset "$WORK/datasets/$NAME.dat" --n-datapoints "$n" --psd "$PSD" ;;
    train)
        mkdir -p "$RUN" && cd "$RUN"
        flow=$(field 3)
        if [ "$flow" != train ]; then
            cp "$WORK/$flow/flow.zip" "$RUN/flow.zip"
            echo "flow of $flow"
        else
            # shellcheck disable=SC2046
            $MBANK_CLI "$MBANK/bin/mbank_train_flow" "$INI" --run-dir "$RUN/" \
                --dataset "$WORK/datasets/$(field 2).dat" --flow-file "$RUN/flow.zip" $(field 4)
        fi ;;
    place)
        cd "$RUN"
        # shellcheck disable=SC2046
        $MBANK_CLI "$MBANK/bin/mbank_place_templates" "$INI" --run-dir "$RUN/" \
            --flow-file "$RUN/flow.zip" --bank-file "$RUN/bank.npy" --psd "$PSD" $(field 5) ;;
    diagnose)
        python "$HERE/diagnose_flow.py" --ini "$INI" --flow "$RUN/flow.zip" \
            --dataset "$WORK/datasets/$(field 2).dat" --out "$RUN/diagnostics.json" ;;
    ff)
        cd "$ISOBANK"
        python scripts/evaluate_bank.py --config configs/eccentric_5d.py --bank "$RUN/bank.npy" \
            --save-ff "$RUN/ff.npy" ;;
    plots)
        python "$HERE/plot_tuning.py" --work "$WORK" --variants "$HERE/variants.tsv" \
            --isobank-dir "$ISOBANK/runs/eccentric_5d" ;;
    *)
        echo "unknown step: $STEP" >&2; exit 1 ;;
esac
