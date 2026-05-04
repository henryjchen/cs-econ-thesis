# Mexico Census Population Density Prediction

Author: Henry Chen
Advisor: Luke Sanford (Yale School of the Environment)
Contact: henry.chen@yale.edu

This project explores socioeconomic estimation (specifically population density) in Mexico using satellite imagery and foundation models. It compares two approaches: **AlphaEarth** (static embeddings) and **CLAY** (dynamic foundation model) for the years 2010 and 2020.

## Project Structure

```text
.
├── alphaearth_src/             # AlphaEarth pipeline (Census cleaning + XGBoost/Ridge)
│   ├── scripts/                # Data prep, modeling, and evaluation scripts
│   ├── notebooks/              # EDA and model evaluation notebooks
│   ├── data/                   # Census CSVs, embeddings, and geospatial files
│   └── outputs/                # Model checkpoints and evaluation metrics
├── clay_src/                   # CLAY foundation model pipeline
│   └── CLAY_mexico_thesis/
│       └── CLAY_Inference_Project/
│           ├── Clay/           # Clay foundation model source & configuration
│           ├── embedding_extract/ # Main pipeline (GEE -> CLAY -> Aggregation)
│           ├── cluster_scripts/ # Slurm/HPC helper scripts
│           ├── data/           # Input shapefiles (AGEBs/Municipios)
│           ├── out/            # (Created at runtime) CLAY 2010 outputs
│           └── out_2020/       # (Created at runtime) CLAY 2020 outputs
├── results_analysis/           # Final cross-model comparison
│   ├── clay2010/               # Artifacts collected from CLAY 2010 run
│   ├── clay2020/               # Artifacts collected from CLAY 2020 run
│   ├── alphaearth2020/         # Artifacts collected from AlphaEarth 2020 run
│   └── compare_results.ipynb   # Final analysis and figure generation
├── google_earth_engine_scripts/ # JavaScript scripts for GEE asset generation
└── README.md                   # This file
```

---

## 1. Preliminary 

Please note that access to an HPC is highly recommended for these steps as well as the CLAY steps. I used Yale's [Bouchet cluster](https://docs.ycrc.yale.edu/clusters/bouchet/).

### Data Preparation
1.  **Download Census Data:** Download the INEGI CSV files and the corresponding shapefiles for the relevant census years (2010, 2020).
2.  **Cleaning:** Clean the CSV by generating the correct `CVEGEO` identifier (preserving leading zeros for Entidad, Municipio, and AGEB). 
3.  **Geospatial Join:** Join the cleaned CSV with the shapefile in your GIS software of choice (QGIS/ArcGIS). 
    *Note: Cleaned CSVs and merged shapefiles are already provided in the repository, so these steps are optional unless starting from scratch.*

### Earth Engine Service Account Setup
Authentication is required to access data from Earth Engine.
1.  Go to the [Google Earth Engine Project Website](https://code.earthengine.google.com/).
2.  Navigate to **Navigation & Menu** → **IAM & Admin** → **Service Accounts**.
3.  Select **Create new account** (or use an Existing Account).
4.  Navigate to **KEYS** → **Add Key** → **Create new key** → **JSON**.
5.  Save the JSON file (e.g., `mexico-key.json`) to your HPC or local project root.
6.  Run the following to persist your credentials:
    ```bash
    echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"' >> ~/.bashrc
    source ~/.bashrc
    ```

### Virtual Environment Setup (CLAY)
A dedicated virtual environment (`claymodel`) is required for the CLAY pipeline.
1.  **Base Installation:** Follow the official instructions [here](https://clay-foundation.github.io/model/getting-started/installation.html).
2.  **Dependencies:** 
    *   Install the Earth Engine API: `pip install earthengine-api`
    *   **Note:** If the provided `environment.yml` fails, manually install the core packages: `pip install torch torchvision torchaudio geopandas shapely pyarrow pandas numpy scikit-learn xgboost matplotlib tqdm`.
3.  **Model Weights:** Download `clay-v1-base.ckpt` from [Hugging Face](https://huggingface.co/made-with-clay/Clay/tree/main) and place it somewhere accessible on your local or HPC system (you will need to reference it later)

---

## 2. AlphaEarth Replication

Execute all commands from the `alphaearth_src/` directory.

1.  **GEE Extraction (Optional):** Run `google_earth_engine_scripts/alphaearth_embedding_extract.js` in the GEE Code Editor. Save the resulting CSV shards to `alphaearth_src/data/embeddings/`.
2.  **Concatenate Embeddings:**
    ```bash
    python scripts/data_prep/concat_embeddings.py
    ```
3.  **Hyperparameter Tuning:**
    ```bash
    python scripts/modeling/tune_parameters.py
    ```
4.  **Train Models:**
    ```bash
    python scripts/modeling/train_models.py
    ```
5.  **Evaluate:**
    ```bash
    python scripts/evaluation/municipio_evaluation.py
    ```
6.  **Export Artifacts:**
    ```bash
    python scripts/evaluation/export_results_analysis_artifacts.py
    # This populates results_analysis/alphaearth2020/
    ```

---

## 3. CLAY Replication (HPC/Slurm)

Navigate to the project root: `cd clay_src/CLAY_mexico_thesis/CLAY_Inference_Project/`. I use 2020 as a placeholder, but feel free to repeat for 2010 replication as well.

### Stage 1: Imagery & Configuration
1.  **Generate Composites:** Run `google_earth_engine_scripts/generate_clay_cloud_composite.js` on GEE for 2010 and 2020. Save these as Assets on GEE.
2.  **Update Config:** In `embedding_extract/config.py`:
    *   Set `YEARS[2010].gee_asset_id` and `YEARS[2020].gee_asset_id` to your respective GEE asset paths.
    *   Update `WEIGHTS_PATH` to point to the location of your `clay-v1-base.ckpt` file.

### Stage 2: Normalization Statistics
This step ensures satellite imagery is correctly scaled for the foundation model.
*   **Slurm:** `sbatch cluster_scripts/compute_feature_norm_stats.sh --year 2020 --n-chunks 5000`
*   **Python:** `python -m embedding_extract.diagnostics.compute_norm_stats --year 2020 --n-chunks 5000`

### Stage 3: Embedding Extraction (Requires GPU)
Downloads imagery from GEE and processes it through the Clay encoder.
*   **Slurm:** `sbatch cluster_scripts/extract_feature_embeddings.sh --year 2020`
*   **Python:** `python -m embedding_extract.pipeline.extract_embeddings --year 2020`

### Stage 4: Spatial Aggregation
Assigns the 480m cell embeddings to AGEB and Municipio polygons.
*   **Slurm:** `sbatch cluster_scripts/aggregate_ageb_features.sh --year 2020`
*   **Python (AGEB):** `python -m embedding_extract.pipeline.aggregate_to_polygons --year 2020`
*   **Python (Muni):** `python -m embedding_extract.pipeline.aggregate_to_municipios --year 2020`

### Stage 5: Hyperparameter Tuning
Finds the optimal Ridge alpha and XGBoost parameters for the generated embeddings.
*   **Slurm:** `sbatch cluster_scripts/tune_best_params.sh --year 2020`
*   **Python:** `python -m embedding_extract.evaluation.tune --year 2020 --drop-zero-target`

### Stage 6: Final ML Analysis
Fits the final models (Ridge + XGBoost) across all spatial scales and generates evaluation figures.
*   **Slurm:** `sbatch cluster_scripts/run_all_analyses_year.sh --year 2020`
*   **Python:** `./embedding_extract/run_all_analyses.sh --year 2020`

### Stage 7: Cross-Temporal Analysis
Evaluates model transferability and decadal change by linking 2010 and 2020 data.
*   **Slurm:** `sbatch cluster_scripts/cross_temporal_analysis.sh --drop-zero-target`
*   **Python:** `python -m embedding_extract.evaluation.cross_temporal --drop-zero-target`

---

## 4. Results Analysis & Comparison

1.  **Gather Artifacts:** Collect the following directories from the CLAY output locations (`out/` for 2010, `out_2020/` for 2020) and move them to `results_analysis/`:
    *   **Required Folders:** `analysis_ageb/`, `analysis_muni_direct/`, `analysis_muni_rollup/`, `cross_temporal/`.
    *   **Final Structure:** `results_analysis/clay2010/` and `results_analysis/clay2020/`.
2.  **Generate Comparative Figures:**
    Open `results_analysis/compare_results.ipynb` and execute all cells. This notebook performs the final performance comparison between the AlphaEarth and CLAY methodologies.
