#!/usr/bin/env bash
# Run the full analysis pipeline across all three aggregation scales in one
# shot. For each scale we do two things:
#   1. analysis.py  -- expensive: fits Ridge + XGBoost, writes metrics/JSON/CSV
#                      and a predictions_oof_*.parquet.
#   2. plots.py     -- cheap:     re-reads those artifacts and renders every
#                      report-ready figure.
#
# Splitting the two steps lets you iterate on figure captions and layouts by
# calling plots.py on its own (see ``--skip-analysis`` below) without having
# to re-fit XGBoost.
#
# The ``--year`` flag selects which dataset to run against (2010 or 2020)
# and is forwarded to every underlying Python call. Outputs for 2010 live
# under embedding_extract/out/; outputs for 2020 live under
# embedding_extract/out_2020/, so 2010 and 2020 artifacts never
# overwrite each other.
#
# Usage:
#     ./embedding_extract/run_all_analyses.sh                    # 2010 (default)
#     ./embedding_extract/run_all_analyses.sh --year 2020        # 2020
#     ./embedding_extract/run_all_analyses.sh --ridge-alpha 100
#     ./embedding_extract/run_all_analyses.sh --year 2020 --ridge-alpha 100
#     ./embedding_extract/run_all_analyses.sh --target log_POBTOT --skip-xgb
#     ./embedding_extract/run_all_analyses.sh --skip-analysis    # replot only
#
# Assumes you've already run (for the same --year):
#     python -m embedding_extract.pipeline.extract_embeddings             [--year 2020]
#     python -m embedding_extract.pipeline.aggregate_to_polygons          [--year 2020]
#     python -m embedding_extract.pipeline.aggregate_to_municipios        [--year 2020]  # rollup
#     python -m embedding_extract.pipeline.aggregate_to_municipios_direct [--year 2020]  # dissolve

set -euo pipefail

# Resolve to the project root (parent of embedding_extract/), regardless
# of where the caller invokes the script from.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
cd "$PROJECT_ROOT"

# Peel off ``--skip-analysis`` and ``--year <yyyy>`` so we know which out-root
# to write to; everything else is forwarded verbatim to analysis.py (which
# will itself re-parse --year from EXTRA_ARGS if present).
SKIP_ANALYSIS=0
YEAR=2010
EXTRA_ARGS=()

while (( "$#" )); do
    case "$1" in
        --skip-analysis)
            SKIP_ANALYSIS=1
            shift
            ;;
        --year)
            YEAR="$2"
            EXTRA_ARGS+=("--year" "$2")
            shift 2
            ;;
        --year=*)
            YEAR="${1#*=}"
            EXTRA_ARGS+=("$1")
            shift
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$YEAR" in
    2010) OUT="embedding_extract/out" ;;
    2020) OUT="embedding_extract/out_2020" ;;
    *)
        echo "Unknown --year=$YEAR. Supported years: 2010, 2020." >&2
        exit 2
        ;;
esac

echo "Year      : $YEAR"
echo "Out root  : $OUT"
echo "Extras    : ${EXTRA_ARGS[*]:-<none>}"
echo "Step      : $([[ "$SKIP_ANALYSIS" == 1 ]] && echo 'plots only' || echo 'analysis + plots')"

run_one () {
    local name="$1"       # "ageb" | "muni_rollup" | "muni_direct"
    local parquet="$2"    # input parquet path
    local out_dir="$3"    # destination folder for metrics + plots

    echo ""
    echo "=============================================================="
    echo " [$name]  parquet : $parquet"
    echo "          out_dir : $out_dir"
    echo "=============================================================="

    if [[ ! -f "$parquet" ]]; then
        echo "  SKIP: $parquet not found. Generate it first (see header)." >&2
        return 0
    fi

    mkdir -p "$out_dir"

    if [[ "$SKIP_ANALYSIS" == 0 ]]; then
        python3 -u -m embedding_extract.evaluation.analysis \
            --parquet "$parquet" \
            --out-dir "$out_dir" \
            "${EXTRA_ARGS[@]}"
    fi

    # Figures always rebuild from whatever artifacts are in --out-dir.
    # We still forward --year so plots.py picks the right shapefile for the
    # state choropleth. ``--out-dir`` wins over ``--year`` for artifact
    # discovery, which is what we want here.
    python3 -u -m embedding_extract.evaluation.plots \
        --year "$YEAR" \
        --out-dir "$out_dir"
}

run_one "ageb"        "$OUT/ageb_features.parquet"              "$OUT/analysis_ageb"
run_one "muni_rollup" "$OUT/municipio_features.parquet"         "$OUT/analysis_muni_rollup"
run_one "muni_direct" "$OUT/municipio_features_direct.parquet"  "$OUT/analysis_muni_direct"

echo ""
echo "=============================================================="
echo " All runs complete (year=$YEAR). Results under:"
echo "   $OUT/analysis_ageb/"
echo "   $OUT/analysis_muni_rollup/"
echo "   $OUT/analysis_muni_direct/"
echo ""
echo " To re-render figures without re-running models:"
echo "   ./embedding_extract/run_all_analyses.sh --year $YEAR --skip-analysis"
echo "=============================================================="
