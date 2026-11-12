#!/bin/bash
#SBATCH --job-name=svd_grid_occupancy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Array index 0 = ecc5d, 1 = al3d. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
SPACES=(ecc5d al3d)
python -u svd_grid_occupancy.py ${SPACES[$SLURM_ARRAY_TASK_ID]}
