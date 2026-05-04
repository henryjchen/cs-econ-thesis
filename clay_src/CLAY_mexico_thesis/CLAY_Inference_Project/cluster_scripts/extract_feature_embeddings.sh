#!/bin/bash
#SBATCH --partition=gpu                                                    # GPU partition (CLAY encoder runs much faster with CUDA)
#SBATCH --job-name=ft_extract_embed                                        # Job name
#SBATCH --ntasks=1                                                         # Number of tasks
#SBATCH --nodes=1                                                          # Number of nodes
#SBATCH --cpus-per-task=8                                                  # CPU cores (GEE downloads benefit from some parallelism)
#SBATCH --mem=64G                                                          # Memory per node
#SBATCH --gres=gpu:1                                                       # One GPU is enough for single-patch inference
#SBATCH --time=24:00:00                                                    # 24h cap; many chunks + GEE latency
#SBATCH --output=cluster_logs/ft_extract_embed_%j.log                      # stdout log
#SBATCH --error=cluster_logs/ft_extract_embed_%j.err                       # stderr log

set -x

#################################################
# please DO NOT remove the following two commands
#################################################
module load StdEnv
module load miniconda
source $(conda info --base)/etc/profile.d/conda.sh
export SLURM_EXPORT_ENV=ALL

conda activate claymodel

#################################################

# Project root (matches download_mexico.sh / train_mexico_cv_fold.sh)
cd /nfs/roberts/project/pi_ls2375/hjc43/CLAY_mexico_thesis/CLAY_Inference_Project

# Make sure Python picks up ``embedding_extract`` as a top-level package
# even when the module-style invocation is skipped.
export PYTHONPATH="$PWD:${PYTHONPATH}"

echo "Starting Feature Extraction (CLAY embeddings over AGEBs): ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "GPUs: ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"

# Run the extractor (writes embedding_cells.parquet + embeddings.npy under
# the per-year output dir selected by ``--year``; default is 2010). Forward
# any extra args to the Python module so you can submit either year with:
#   sbatch cluster_scripts/extract_feature_embeddings.sh
#   sbatch cluster_scripts/extract_feature_embeddings.sh --year 2020
time python3 -u -m embedding_extract.pipeline.extract_embeddings "$@"
