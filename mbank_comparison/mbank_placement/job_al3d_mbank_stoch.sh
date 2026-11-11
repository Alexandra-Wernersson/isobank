#!/bin/bash
#SBATCH --job-name=al3d_mbank_stoch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH -t 12:00:00
#SBATCH --output=/home/awernersson/projects/mbank/slurm_outputs/%x-%j-%N_slurm.out
source /home/awernersson/venvs/tmnre/bin/activate
export PYTHONPATH=/home/awernersson/projects/mbank:$PYTHONPATH
cd /home/awernersson/projects/mbank
python place_al3d_mbank_stochastic.py
