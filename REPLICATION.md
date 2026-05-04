Replication instructions

Preliminary Steps
1. (Optional since I already provide the cleaned csvs) data prep: download the inegi csv files for relevant census year. download the shapefile for that year too. clean the csv by creating the write CVEGEO figure. join the csv with shape in your GIS software of choice.
2. environmental setup of GEE project. set up service account and get keys.
3. set up  clay 
    """To get started, you will need a virtual environment set up with the necessary dependencies. Follow Claymodel to get set up.

    NOTE: environment.yml in model may not be configured correctly at this point in time. If you have issues installing the virtual environment, contact one of us.
    To activate claymodel, run conda activate claymodel.

    Then, you will need to install the Earth Engine API Library. Run pip install earthengine-api while claymodel is activated.

    Authentication will be required to access data from Earth Engine. After installing, you will need to download a JSON file containing your permissions. Go to your Earth Engine Project Website, click Navigation & Menu → IAM & Admin → Service Accounts → Create new account OR Existing Account → KEYS → Add Key via JSON file. From there, save the JSON file somewhere in Grace and run the following command:
    echo 'export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your-key.json"' >> ~/.bashrc
    source ~/.bashrc
    Now, you can run the Python programs and bash scripts in cluster_scripts/"""


AlphaEarth Replication
1. (optional) run the alphaearth_embedding_extract.js code on GEE and save the embedding outputs to alphaearth_src/data/embeddings folder. process all the embeddings and concat them into one csv. 
2. cd into alphaearth_src folder and run python scripts/modeling/tune_parameters.py
3. next run python scripts/modeling/train_models.py
4. run python scripts/evaluation/municipio_evaluation.py
5. finally run scripts/evaluation/export_results_analysis_artifacts to get the csvs and other parquet files saved into the ./results_analysis file, which is what is used to actually produce the figures from this project (see their instructions for more)


Clay replication
Please note you need an HPC for this. I used Yale's Bouchet cluster and I will try to specify what parittion i used for each step (otherwise, please refer to the code base...)

1. go to GEE and run the "generate_cloud_composite.js" script for each period you are intersted in 2010 and 2020. Save them as an asset on GEE.
2. ssh into cluster and cd to clay_src/clay_mexico_thesis/clay_inference_project. in embedding_extract/config.py update gee_asset_id to your two cloud composite asset ids and WEIGHTS_PATH to whereever the weights are stored on your HPC. if not already on hpc, you can downlaod them here https://huggingface.co/made-with-clay/Clay/tree/main
3. preprocessing:   Compute the normalization statistics for the 2020 imagery. This ensures the satellite data is scaled correctly for the foundation model.
   * Slurm: sbatch cluster_scripts/compute_feature_norm_stats.sh --year 2020 --n-chunks 5000
   * Python: python -m embedding_extract.diagnostics.compute_norm_stats --year 2020 --n-chunks 5000
4.  embedding pipeline
  This stage converts raw satellite imagery into high-dimensional vectors and assigns them to census boundaries.
  Step 1: Feature Extraction (Requires GPU)
  This script downloads imagery from GEE, passes it through Clay, and saves .npy embedding files.
   * Slurm: sbatch cluster_scripts/extract_feature_embeddings.sh --year 2020
   * Python: python -m embedding_extract.pipeline.extract_embeddings --year 2020
5. Spatial Aggregation
  Assign the 480m cell embeddings to AGEB and Municipio polygons.
   * Slurm: sbatch cluster_scripts/aggregate_ageb_features.sh --year 2020
   * Python (AGEB): python -m embedding_extract.pipeline.aggregate_to_polygons --year 2020
   * Python (Muni): python -m embedding_extract.pipeline.aggregate_to_municipios --year 202
6.  Step 1: Hyperparameter Tuning
  Find the optimal Ridge alpha and XGBoost settings for the 2020 data.
   * Slurm: sbatch cluster_scripts/tune_best_params.sh --year 2020
   * Python: python -m embedding_extract.evaluation.tune --year 2020 --drop-zero-target
7.  Fits the final models (Ridge + XGBoost) across all spatial scales and generates all figures.
   * Slurm: sbatch cluster_scripts/run_all_analyzses_year.sh --year 2020
   * Python: ./embedding_extract/run_all_analyses.sh --year 2020


results_analysis replication
1. download/scp/move the following folders from out and out_2020/ in 	analysis_ageb, analysis_muni_direct,	analysis_muni_rollup, cross_temporal. put them in folders titled "clay2010" and "clay2020" respectiveully. (also ensure the alphaearth2020 folder exists and is in tact)
2. run compare_results.ipynb cells
