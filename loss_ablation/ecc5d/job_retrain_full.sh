#!/bin/bash
#SBATCH --job-name=abl_retrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	Retraining variance of the full objective: 0-1 new initialization seeds on the shared pairs,
#	2-4 redrawn pairs (pair seed 1-3, torch seed 101-103). Needs job_pairs.sh and job_retrain_pairs.sh first.
i=$SLURM_ARRAY_TASK_ID
if [ $i -lt 2 ]; then python -u train_arm.py full $i
else P=$((i - 1)); python -u train_arm.py full $((100 + P)) p=$P; fi
