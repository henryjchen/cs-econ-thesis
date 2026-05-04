#!/bin/bash
# From project root once: mkdir -p cluster_logs  (Slurm needs this path at submit time)
#SBATCH --partition=cpu
#SBATCH --job-name=ft_analyze
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_logs/ft_analyze_%j.log
#SBATCH --error=cluster_logs/ft_analyze_%j.err

# Ridge + XGBoost analysis on ageb_features.parquet.
# Reports overall R^2 and per-state R^2 (CVEGEO first 2 chars).
#
# Pass ``--year 2020`` to run against 2020 data (selects both the input
# parquet and the default output dir). Example:
#   sbatch cluster_scripts/analyze_ageb_features.sh
#   sbatch cluster_scripts/analyze_ageb_features.sh --year 2020 --ridge-alpha 100

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

echo "Starting analysis: ${SLURM_JOBID}"
echo "Node: $(hostname -s)"
echo "CPUs: ${SLURM_CPUS_ON_NODE}"

# Step 1 -- expensive: fit Ridge + XGBoost, write JSON/CSV/parquet artifacts.
time python3 -u -m embedding_extract.evaluation.analysis "$@"

# Step 2 -- cheap: re-read those artifacts and render the figure set. Kept
# separate so figures can also be regenerated locally (without resubmitting)
# via: python -m embedding_extract.evaluation.plots --out-dir <dir>
#
# We can't just forward "$@" because plots.py has a narrower arg surface; we
# only need to point it at the same --out-dir analysis used, and pass --year
# so the state choropleth pulls the right shapefile. Parse both out.
OUT_DIR=""
YEAR=""
prev=""
for arg in "$@"; do
    case "$prev" in
        --out-dir) OUT_DIR="$arg" ;;
        --year)    YEAR="$arg"    ;;
    esac
    case "$arg" in
        --out-dir=*) OUT_DIR="${arg#*=}" ;;
        --year=*)    YEAR="${arg#*=}"    ;;
    esac
    prev="$arg"
done

PLOT_ARGS=()
[[ -n "$OUT_DIR" ]] && PLOT_ARGS+=("--out-dir" "$OUT_DIR")
[[ -n "$YEAR"    ]] && PLOT_ARGS+=("--year"    "$YEAR")

time python3 -u -m embedding_extract.evaluation.plots "${PLOT_ARGS[@]}"
