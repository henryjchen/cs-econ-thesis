#!/bin/bash
# Compute standalone embedding_extract norm stats from GEE patch samples.
#
# Examples:
#   sbatch cluster_scripts/compute_feature_norm_stats.sh --year 2010
#   sbatch cluster_scripts/compute_feature_norm_stats.sh --year 2020 --n-chunks 5000
#   sbatch cluster_scripts/compute_feature_norm_stats.sh --year 2020 --all-chunks
#
#SBATCH --partition=day
#SBATCH --job-name=ft_norm_stats
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/ft_norm_stats_%j.log
#SBATCH --error=cluster_logs/ft_norm_stats_%j.err

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

echo "Starting feature norm stats: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Args: $*"

time python3 -u -m embedding_extract.diagnostics.compute_norm_stats "$@"

