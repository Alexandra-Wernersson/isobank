#!/bin/bash
#SBATCH --job-name=abl_retrain_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:30:00
#SBATCH --array=1-3
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	independently redrawn training pairs for the retraining-variance runs (pair seeds 1-3)
python -u make_pairs.py train p=$SLURM_ARRAY_TASK_ID
