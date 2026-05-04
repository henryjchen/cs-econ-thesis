#!/bin/bash
#SBATCH --partition=day
#SBATCH --job-name=ft_agg_ageb
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=cluster_logs/ft_agg_ageb_%j.log
#SBATCH --error=cluster_logs/ft_agg_ageb_%j.err

# Aggregate cell embeddings → one vector per AGEB (no GPU, no Earth Engine).
# Expects embedding_cells.parquet + embeddings.npy under the per-year output
# dir from extract_feature_embeddings.sh. Writes ageb_features.parquet to
# the same dir. Pass ``--year 2020`` to run against 2020 data:
#   sbatch cluster_scripts/aggregate_ageb_features.sh
#   sbatch cluster_scripts/aggregate_ageb_features.sh --year 2020

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

# Project root (matches extract_feature_embeddings.sh)
cd /nfs/roberts/project/pi_ls2375/hjc43/CLAY_mexico_thesis/CLAY_Inference_Project


export PYTHONPATH="$PWD:${PYTHONPATH}"

echo "Starting AGEB aggregation: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"

time python3 -u -m embedding_extract.pipeline.aggregate_to_polygons "$@"
