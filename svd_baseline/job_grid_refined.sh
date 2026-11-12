#!/bin/bash
#SBATCH --job-name=svd_grid_refined
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH --output=logs/%x-%j.out
#	Submit from this folder. Eccentric 5D only. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$SLURM_SUBMIT_DIR"
python -u svd_grid_refined.py ecc5d
