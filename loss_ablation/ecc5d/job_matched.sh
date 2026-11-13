#!/bin/bash
#SBATCH --job-name=abl_matched
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH -t 04:00:00
#SBATCH --array=0-38
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	For each trained flow (index into RUN_LIST in place_matched.py), search the placement radius for a bank of
#	2,267,558 templates (+/-3%), then score it on 4,000 injections (seed 4002, 200 nearest templates, exact matches).
#	Indices: 0-6 arms seed 0, 7-13 arms seed 1 (order of ARMS in common.py), 14-16 full objective on redrawn pairs,
#	17-18 loc_edge, 19-20 no_bound seeds 2-3, 21-22 no_tail seeds 2-3, 23-38 weight scan.
python -u place_matched.py $SLURM_ARRAY_TASK_ID
