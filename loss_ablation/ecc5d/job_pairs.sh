#!/bin/bash
#SBATCH --job-name=abl_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	shared training pairs (used by every run with pair seed 0) and held-out test pairs
python -u make_pairs.py both
