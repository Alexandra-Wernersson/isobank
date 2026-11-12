#!/bin/bash
#SBATCH --job-name=svd_feasibility
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 00:40:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Array index 0 = ecc5d, 1 = al3d. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
SPACES=(ecc5d al3d)
python -u svd_feasibility.py ${SPACES[$SLURM_ARRAY_TASK_ID]}
