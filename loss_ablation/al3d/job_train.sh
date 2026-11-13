#!/bin/bash
#SBATCH --job-name=al3d_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 01:50:00
#SBATCH --array=0-2,4-8
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	0: full s0, 1-2: no_global s0-1, 4-6: full s1-3, 7-8: no_global s2-3 (3 is the paper flow, not trained here)
python -u al3d_ablation.py train $SLURM_ARRAY_TASK_ID
