#!/bin/bash
# Run cross-temporal 2010 -> 2020 evaluation:
# - train 2010 embeddings/labels, predict 2020 labels
# - matched AGEB 2010 embeddings -> 2020 labels
# - matched AGEB [2010 embeddings, 2020-2010 delta] -> 2020 labels
#
# Uses tuned params from embedding_extract/evaluation/best_params.json
# when available.
#
# Example:
#   sbatch cluster_scripts/cross_temporal_analysis.sh --drop-zero-target
#   sbatch cluster_scripts/cross_temporal_analysis.sh --drop-zero-target --skip-xgb
#
#SBATCH --partition=day
#SBATCH --job-name=ft_cross_time
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=cluster_logs/ft_cross_time_%j.log
#SBATCH --error=cluster_logs/ft_cross_time_%j.err

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

echo "Starting cross-temporal analysis: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Args: $*"

time python3 -u -m embedding_extract.evaluation.cross_temporal "$@"

