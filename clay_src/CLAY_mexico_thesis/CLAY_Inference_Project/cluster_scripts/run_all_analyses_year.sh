#!/bin/bash
# Run all three evaluation layers for a specified year:
# - AGEB
# - municipio rollup
# - municipio direct
#
# This uses `analysis.py` (which auto-loads tuned hyperparameters from
# `embedding_extract/evaluation/best_params.json` by default) and always
# drops zero-population rows via `--drop-zero-target`.
#
# Submit examples:
#   sbatch cluster_scripts/run_all_analyses_year.sh --year 2010
#   sbatch cluster_scripts/run_all_analyses_year.sh --year 2020
#
# You can pass additional `analysis.py` flags too, e.g.:
#   sbatch cluster_scripts/run_all_analyses_year.sh --year 2020 --target log_POBTOT
#
#SBATCH --partition=day
#SBATCH --job-name=ft_all_year
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/ft_all_year_%j.log
#SBATCH --error=cluster_logs/ft_all_year_%j.err

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

# Make sure Python picks up `embedding_extract` as a top-level package.
export PYTHONPATH="$PWD:${PYTHONPATH}"

echo "Starting run_all_analyses (year from args): ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Args: $*"

time bash embedding_extract/run_all_analyses.sh \
  "$@" \
  --drop-zero-target

