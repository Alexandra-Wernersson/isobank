#!/bin/bash
#SBATCH --job-name=al3d_pairs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:40:00
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
python -u al3d_ablation.py pairs
