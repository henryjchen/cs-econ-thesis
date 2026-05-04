#!/bin/bash
# Visualize a small sample of live GEE input chips and their CLAY token embeddings.
# Outputs land under viz_chips/chip_<id>/inputs/*.png and .../embeddings/*.png by default.
#
# Examples:
#   sbatch cluster_scripts/visualize_feature_chips.sh --n-chips 4
#   sbatch cluster_scripts/visualize_feature_chips.sh --year 2020 --n-chips 8 --seed 0
#   sbatch cluster_scripts/visualize_feature_chips.sh --output-dir embedding_extract/out/debug_viz
#
#SBATCH --partition=gpu
#SBATCH --job-name=ft_viz_chips
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=cluster_logs/ft_viz_chips_%j.log
#SBATCH --error=cluster_logs/ft_viz_chips_%j.err

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

echo "Starting feature chip visualization: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "GPUs: ${SLURM_STEP_GPUS:-$SLURM_JOB_GPUS}"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"
echo "Args: $*"

time python3 -u -m embedding_extract.diagnostics.visualize_chips "$@"
