#!/bin/bash
#SBATCH --job-name=abl_arms
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH -t 03:00:00
#SBATCH --array=0-33
#SBATCH --output=logs/%x-%A_%a.out
#	Submit from this folder. Set ISOBANK_ROOT to the isobank release if it is not at the default path. Activate your environment below.
# source /path/to/venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"
#	Ablated and reweighted objectives (arm and torch seed; arms are defined in common.py), all on the shared pairs.
RUNS=("no_global 0" "no_global 1" "no_local 0" "no_local 1" "no_bound 0" "no_bound 1" "no_bound 2" "no_bound 3"
      "no_edge 0" "no_edge 1" "global_only 0" "global_only 1" "no_tail 0" "no_tail 1" "no_tail 2" "no_tail 3"
      "loc_edge 0" "loc_edge 1"
      "loc_x0p5 0" "loc_x0p5 1" "loc_x2 0" "loc_x2 1" "bnd_x0p5 0" "bnd_x0p5 1" "bnd_x2 0" "bnd_x2 1"
      "edge_x0p5 0" "edge_x0p5 1" "edge_x2 0" "edge_x2 1" "tail_1 0" "tail_1 1" "tail_10 0" "tail_10 1")
python -u train_arm.py ${RUNS[$SLURM_ARRAY_TASK_ID]}
