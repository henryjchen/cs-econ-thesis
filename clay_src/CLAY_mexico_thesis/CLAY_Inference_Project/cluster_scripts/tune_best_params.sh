#!/bin/bash
# Tune Ridge + XGBoost hyperparameters (writes/updates best_params.json).
#
# Example:
#   sbatch cluster_scripts/tune_best_params.sh --year 2010 --drop-zero-target
#   sbatch cluster_scripts/tune_best_params.sh --year 2020 --drop-zero-target
#
# Notes:
# - Runtime is dominated by XGBoost random search (default 100 trials).
# - This script forwards "$@" to the tuning CLI, so you can override:
#     --xgb-trials 200  --xgb-rounds 3000  --xgb-early-stop 100  --target log_POBTOT
#
#SBATCH --partition=day
#SBATCH --job-name=ft_tune
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=cluster_logs/ft_tune_%j.log
#SBATCH --error=cluster_logs/ft_tune_%j.err

set -e
set -x

#################################################
# please DO NOT remove the following two commands
#################################################
module load StdEnv
module load miniconda
source "$(conda info --base)/etc/profile.d/conda.sh"
export SLURM_EXPORT_ENV=ALL

conda activate claymodel

#################################################

cd /nfs/roberts/project/pi_ls2375/hjc43/CLAY_mexico_thesis/CLAY_Inference_Project

mkdir -p cluster_logs
export PYTHONPATH="$PWD:${PYTHONPATH}"

echo "Starting tuning: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Args: $*"

time python3 -u -m embedding_extract.evaluation.tune "$@"

