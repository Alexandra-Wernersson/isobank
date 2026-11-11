#!/bin/bash
#SBATCH --job-name=ecc5D_mbank_nat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --partition=rome
#SBATCH -t 24:00:00
#SBATCH --output=/home/awernersson/projects/mbank/slurm_outputs/%x-%j-%N_slurm.out
source /home/awernersson/venvs/tmnre/bin/activate
export PYTHONPATH=/home/awernersson/projects/mbank:$PYTHONPATH
cd /home/awernersson/projects/mbank

python run_mbank_cli.py bin/mbank_generate_flow_dataset ecc5D_mbank_native.ini
python run_mbank_cli.py bin/mbank_train_flow ecc5D_mbank_native.ini
python run_mbank_cli.py bin/mbank_place_templates ecc5D_mbank_native.ini
