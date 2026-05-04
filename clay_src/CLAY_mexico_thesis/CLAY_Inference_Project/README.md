# CLAY Inference Project

Remote-sensing workflows for socioeconomic estimation using the [Clay](https://clay-foundation.github.io/model/) foundation model. This repository contains the **Mexico Feature Extraction** pipeline, which processes Earth Engine imagery through Clay to produce high-dimensional embeddings for AGEB and Municipio polygons.

---

## Project Structure

```
CLAY_Inference_Project/
├── Clay/                 # Clay foundation model source & configs
├── embedding_extract/
│   ├── pipeline/         # GEE -> CLAY -> Embeddings -> Polygons
│   ├── evaluation/       # Downstream ML (Ridge, XGBoost) & Plots
│   ├── diagnostics/      # Progress tracking & visualization
│   ├── core/             # Shared library code
│   └── data/             # Input shapefiles (e.g., AGEBs)
├── cluster_scripts/      # Slurm/HPC helper scripts
└── README.md
```

### Core Components

- **`Clay/`** — Implementation of the Clay foundation model. Requires weights (`clay-v1-base.ckpt`) to be placed in `Clay/weights/` (download from [Hugging Face](https://huggingface.co/made-with-clay/Clay/tree/main)).
- **`embedding_extract/`** — The primary pipeline for Mexico. It extracts 240m resolution embeddings, pools them to 480m cells, and aggregates them into AGEB and Municipio features.

---

## Getting Started

### 1. Setup
- **Python Env:** Follow the [Clay installation guide](https://clay-foundation.github.io/model/getting-started/installation.html).
- **Earth Engine:** Place your service-account JSON at `mexico-key.json` in the project root.
- **Paths:** Configure `embedding_extract/config.py` with your local paths for weights and data.

### 2. Workflow
The pipeline supports multiple years (e.g., `--year 2010` or `2020`) via a unified CLI. Always run scripts from the **project root**.

#### Extraction & Aggregation (HPC/GPU)
```bash
# 1. Extract embeddings (requires GPU)
python -m embedding_extract.pipeline.extract_embeddings --year 2010

# 2. Aggregate to AGEBs
python -m embedding_extract.pipeline.aggregate_to_polygons --year 2010

# 3. Roll up to Municipios
python -m embedding_extract.pipeline.aggregate_to_municipios --year 2010
```

#### Downstream Analysis & Evaluation
```bash
# Run full ML evaluation (Ridge + XGBoost) and generate plots
./embedding_extract/run_all_analyses.sh --year 2010

# Re-render plots only (cheap)
./embedding_extract/run_all_analyses.sh --year 2010 --skip-analysis
```

---

## Key Features

- **Multi-Year Support:** Seamlessly switch between 2010 and 2020 datasets.
- **Spatial Scenarios:** Handles small polygons via point-in-cell assignment (Scenario A) and large polygons via mean-pooling (Scenario B).
- **HPC Optimized:** Batch extraction scripts for Slurm are located in `cluster_scripts/`.
- **Diagnostics:** Use `track_progress` to monitor extraction and `visualize_chips` to inspect raw imagery and embeddings.

For detailed documentation on the pipeline internals, see [embedding_extract/README.md](embedding_extract/README.md).
