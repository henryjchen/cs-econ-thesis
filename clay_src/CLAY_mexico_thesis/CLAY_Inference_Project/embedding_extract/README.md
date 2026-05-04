# Feature extraction: per-AGEB CLAY embeddings

Self-contained module that extracts 16x16 (480 m per token) CLAY embeddings
over the AGEB shapefile and aggregates to one 768-dim vector per polygon.
The only project dependency is the top-level `Clay/` package.

## Layout

```
embedding_extract/
├── config.py                 central config (paths, CRS, CLAY weights, GEE)
├── run_all_analyses.sh       top-level entry: runs evaluation on all 3 scales
├── data/                     raw shapefile inputs (.shp/.dbf/.shx/.prj/.cpg)
├── core/                     shared library code (no CLI)
│   ├── clay_utils.py         CLAY encoder loader + datacube primitives
│   └── grid_utils.py         480 m cell grid + 16x16 chunking over AGEB bounds
├── pipeline/                 main pipeline stages (each is a runnable CLI)
│   ├── extract_embeddings.py          GEE -> CLAY -> embeddings.npy + cells.parquet
│   ├── aggregate_to_polygons.py       AGEB-level (scenario A / B)
│   ├── aggregate_to_municipios.py     municipio rollup from AGEB parquet
│   └── aggregate_to_municipios_direct.py
│                              municipio by dissolving AGEB polygons
├── diagnostics/              validation + progress monitoring
│   ├── track_progress.py              embeddings.npy fill rate
│   └── sanity_check_ageb_features.py  validate ageb_features.parquet
└── evaluation/               downstream ML (two-step, kept separate so you can
    │                          iterate on figures without refitting models)
    ├── analysis.py           expensive: fits Ridge + XGBoost; writes JSON/CSV/parquet
    └── plots.py              cheap:     reads those artifacts; renders PNG figures
```

Outputs land in a per-year directory (gitignored):

- `embedding_extract/out/`      &nbsp;&nbsp;for `--year 2010` (default)
- `embedding_extract/out_2020/` for `--year 2020`

The mapping (shapefile + GEE asset + out-dir per year) lives in
`config.YEARS`; add a new entry there to support another year.

## Requirements

On the cluster (for extraction):
- `geopandas`, `shapely`, `pyproj`, `pyarrow`, `earthengine-api`, `requests`,
  `torch`, `box`, `pyyaml`, `google-auth`
- Earth Engine service-account JSON at
  `config.GOOGLE_APPLICATION_CREDENTIALS_PATH`
  (defaults to `CLAY_Inference_Project/mexico-key.json`).
- Clay checkpoint at `config.WEIGHTS_PATH`.
- (Optional) per-band mean/std `.pth` with keys `mean`/`std` at
  `config.NORM_STATS_PATH`. Set to `None` to skip normalization.

Aggregation, diagnostics, and evaluation are local only -- no Earth Engine
or Clay needed -- just `geopandas`, `shapely`, `pyarrow`, `pandas`, `numpy`,
`scikit-learn`, `xgboost`, `matplotlib`, `tqdm`.

## Usage

Always run from the **project root** (`CLAY_Inference_Project/`). All modules
use the `python -m ...` dotted-path convention.

Every CLI accepts a single `--year` flag (default `2010`) that selects
the shapefile, GEE asset, and per-year output root in lockstep; omit it
for 2010, pass `--year 2020` for 2020. The year registry is in
`embedding_extract/config.py` (see `YEARS`).

```bash
# Stage 1 -- extract 240 m CLAY embeddings over Mexico (GPU cluster).
python -m embedding_extract.pipeline.extract_embeddings
python -m embedding_extract.pipeline.extract_embeddings --year 2020

# (optional) check fill progress between extraction passes.
python -m embedding_extract.diagnostics.track_progress [--year 2020]

# (optional) visualize a few live GEE input chips and their CLAY token embeddings.
# Writes ``viz_chips/chip_<id>/inputs/*.png`` and ``.../embeddings/*.png``.
python -m embedding_extract.diagnostics.visualize_chips --n-chips 4 [--year 2020]
sbatch cluster_scripts/visualize_feature_chips.sh --n-chips 4 [--year 2020]

# (optional) visualize random K-fold CV assignment on a Mexico map (AGEBs).
python -m embedding_extract.diagnostics.plot_cv_folds_map [--year 2020] [--folds 5] [--seed 0] [--per-fold]

# Stage 2a -- aggregate cell embeddings to AGEB polygons.
python -m embedding_extract.pipeline.aggregate_to_polygons [--year 2020]
python -m embedding_extract.diagnostics.sanity_check_ageb_features [--year 2020]

# Stage 2b -- municipio rollup from the AGEB parquet (cheap, ~seconds).
python -m embedding_extract.pipeline.aggregate_to_municipios [--year 2020]

# Stage 2c -- municipio via dissolving AGEB polygons (re-aggregates cells).
python -m embedding_extract.pipeline.aggregate_to_municipios_direct [--year 2020]

# Stage 3 -- fit Ridge + XGBoost at every scale, then render all figures.
./embedding_extract/run_all_analyses.sh --ridge-alpha 100
./embedding_extract/run_all_analyses.sh --year 2020 --ridge-alpha 100

# Stage 3b -- re-render figures only (cheap; no model refitting).
./embedding_extract/run_all_analyses.sh --skip-analysis
./embedding_extract/run_all_analyses.sh --year 2020 --skip-analysis
```

Results for 2010 land in `out/analysis_{ageb,muni_rollup,muni_direct}/` and
for 2020 in `out_2020/analysis_{ageb,muni_rollup,muni_direct}/`, so the two
runs never overwrite each other.

## Hyperparameter tuning (per year)

Ridge and XGBoost can have different best hyperparameters in 2010 vs 2020, so
`embedding_extract` includes a tuning CLI that searches for good settings
and saves them for reuse in all subsequent `analysis.py` runs.

### 1) Tune once (expensive)

This updates `embedding_extract/evaluation/best_params.json`:

```bash
# 2010
python -m embedding_extract.evaluation.tune --year 2010 --drop-zero-target

# 2020
python -m embedding_extract.evaluation.tune --year 2020 --drop-zero-target
```

By default this uses random 5-fold CV (seed=0), optimizes R², tunes Ridge over
a small alpha grid, and runs a 100-trial random search for XGBoost.

### 2) Run analysis normally (auto-uses tuned params)

If `best_params.json` contains an entry matching `(year, target, CV, feature_set)`,
`analysis.py` will load it automatically unless you pass `--no-best-params`.

```bash
python -m embedding_extract.evaluation.analysis --year 2020 --drop-zero-target
python -m embedding_extract.evaluation.analysis --year 2020 --drop-zero-target --no-best-params
```

CLI overrides win. For example, this forces a custom Ridge alpha:

```bash
python -m embedding_extract.evaluation.analysis --year 2020 --ridge-alpha 10
```

> **2020 placeholders.** The 2020 shapefile path and GEE asset ID in
> `config.YEARS[2020]` are marked with `TODO_2020_...` placeholders. Fill
> them in (and drop the shapefile + sidecars into `data/`) before kicking
> off a real 2020 run.

`analysis.py` is the expensive half of Stage 3: it fits models and writes
`metrics_overall_*.json`, `metrics_by_state_*.csv`, and
`predictions_oof_*.parquet`. `plots.py` is a cheap, standalone CLI that
re-reads those three artifacts and regenerates every figure -- so caption
tweaks, colorbar experiments, or new filters don't require a re-fit:

```bash
# render all runs in the default analysis dir
python -m embedding_extract.evaluation.plots

# render only one feature-set tag, from a custom out-dir
python -m embedding_extract.evaluation.plots \
    --out-dir embedding_extract/out/analysis_muni_rollup \
    --tag emb
```

Slurm equivalents live in `cluster_scripts/` (`extract_feature_embeddings.sh`,
`aggregate_ageb_features.sh`, `analyze_ageb_features.sh`).

## Scenarios (AGEB aggregation)

- **Scenario A** -- polygon area `<` 480 m x 480 m **or** no cell centers land
  inside the polygon => find the cell whose 480 m window contains the
  polygon's representative point; assign that 768-dim vector.
- **Scenario B** -- polygon area `>=` 480 m x 480 m **and** at least one cell
  center falls inside the polygon => element-wise mean of all matching cell
  vectors.

The output `ageb_features.parquet` has one row per AGEB with columns
`ageb_id`, `scenario`, `polygon_area_m2`, any passthrough attributes from the
shapefile (e.g. `POBTOT`, `log_popden`), and `emb_0` ... `emb_767`.

## Target convention

The `log_popden` / `log_POBTOT` columns in the source shapefile use
`log(x+1)` (numpy `log1p`), so `log_popden == 0` marks zero-population
AGEBs (not "1 person/km^2"). The municipio aggregators **recompute** these
labels from summed `POBTOT` and match that convention via `log1p`. Pass
`--drop-zero-target` to `analysis.py` to exclude zero-population rows.

## Notes

- `CELL_SIZE_M = 240`, `PATCH_PIXELS = 128`, `SCALE_M = 30` => each CLAY input
  patch covers 16x16 native 240 m tokens, mean-pooled 2x2 to 480 m cells.
- Chunks whose GEE download fails are logged and left as `NaN` embeddings;
  the aggregator treats any non-finite cell vectors the same as missing.
- Set `FILTER_BY_AGEBS = False` in `config.py` to extract embeddings over the
  entire bounding box instead of only chunks intersecting AGEBs.
