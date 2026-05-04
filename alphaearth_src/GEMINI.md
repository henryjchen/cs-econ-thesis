# Mexico 2020 AlphaEarth Pipeline

This project cleans 2020 Mexican census tables, combines AlphaEarth embeddings, and trains population-density prediction models for the thesis workflow.

## Directory Structure

- `scripts/`: runnable Python scripts grouped by workflow stage.
  - `scripts/data_prep/`: census cleaning, AGEB merging, and embedding CSV concatenation.
  - `scripts/modeling/`: hyperparameter tuning (XGBoost + Ridge) and final fold training.
  - `scripts/evaluation/`: optional aggregated metrics (e.g. municipio rollup/direct evaluation).
- `notebooks/`: exploratory analysis and model evaluation notebooks.
- `data/`: input and intermediate data artifacts.
  - `data/raw/`: original census CSVs.
  - `data/processed/`: cleaned census outputs.
  - `data/embeddings/`: AlphaEarth embedding shards and combined embedding table.
  - `data/geospatial/`: shapefiles, QGIS project files, and geospatial archives.
- `outputs/`: generated artifacts.
  - `outputs/models/`: tuned parameters and trained model checkpoints.
  - `outputs/evaluation/`: municipio rollup/direct metrics (`municipio_evaluation.json`).
  - `outputs/figures/`: generated figures.

## Dependencies

Install the main Python dependencies:

```bash
pip install pandas numpy scipy scikit-learn xgboost shap dataframe_image joblib
```

## Pipeline Commands

Run commands from the project root (`alphaearth_src`).

```bash
python scripts/data_prep/clean_csv_ageb.py
python scripts/data_prep/clean_csv_localities.py
python scripts/data_prep/merge_ageb_csvs.py
python scripts/data_prep/concat_embeddings.py
python scripts/modeling/tune_parameters.py
python scripts/modeling/train_models.py
python scripts/evaluation/municipio_evaluation.py
```

## Outputs

- Cleaned AGEB and locality CSVs are written to `data/processed/`.
- The combined AlphaEarth embedding table is written to `data/embeddings/00_all_mexico_embeddings_combined.csv`.
- Tuned XGBoost and Ridge settings (with CV R²) are written to `outputs/models/best_tune_params.json`.
- Fold-specific model checkpoints are written to `outputs/models/checkpoints/`.
- Municipio evaluation metrics (rollup vs direct) are written to `outputs/evaluation/municipio_evaluation.json` when you run `scripts/evaluation/municipio_evaluation.py` or the municipio section of `notebooks/02_model_evaluation.ipynb`.

## Development Conventions

- Preserve leading zeros in geographic codes (`ENTIDAD`, `MUN`, `LOC`, `AGEB`, `CVEGEO`) by reading them as strings.
- When aggregating from block level (`MZA`), exclude aggregate rows and only sum sub-AGEB units.
- Keep notebooks in `notebooks/`; keep runnable scripts in the workflow-specific `scripts/` subfolders.