import json

with open("results_analysis/compare_results.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        new_source = []
        for line in source:
            # 1. _load_ageb_oof
            if "p = RESULTS_DIR / folder / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line and "def _load_ageb_oof" not in "".join(new_source[-2:]):
                new_source.append("    tag = \"emb\" if \"alphaearth\" in folder else \"emb-state\"\n")
                new_source.append("    p = RESULTS_DIR / folder / \"analysis_ageb\" / f\"predictions_oof_{tag}.parquet\"\n")
            elif "print(f\"[skip] missing: {fold}/analysis_ageb/predictions_oof_emb.parquet\")" in line:
                new_source.append("        tag = \"emb\" if \"alphaearth\" in fold else \"emb-state\"\n")
                new_source.append("        print(f\"[skip] missing: {fold}/analysis_ageb/predictions_oof_{tag}.parquet\")\n")
            # 3. _collect_metrics_tables
            elif "p = RESULTS_DIR / fold / \"analysis_ageb\" / \"metrics_by_state_emb.csv\"" in line and "for _lab, fold in _MAP_SOURCES:" in "".join(source):
                new_source.append("        tag = \"emb\" if \"alphaearth\" in fold else \"emb-state\"\n")
                new_source.append("        p = RESULTS_DIR / fold / \"analysis_ageb\" / f\"metrics_by_state_{tag}.csv\"\n")
            # 4. _compute_state_bias
            elif "pred_path = RESULTS_DIR / fold / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line and "\"\"\"Return per-state mean bias" in "".join(source):
                new_source.append("    tag = \"emb\" if \"alphaearth\" in fold else \"emb-state\"\n")
                new_source.append("    pred_path = RESULTS_DIR / fold / \"analysis_ageb\" / f\"predictions_oof_{tag}.parquet\"\n")
            # 5. _label_fold_polygon_frame
            elif "pred_path = RESULTS_DIR / label_fold / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line:
                new_source.append("    tag = \"emb\" if \"alphaearth\" in label_fold else \"emb-state\"\n")
                new_source.append("    pred_path = RESULTS_DIR / label_fold / \"analysis_ageb\" / f\"predictions_oof_{tag}.parquet\"\n")
            # 6. _block_for_fold
            elif "pred_path = RESULTS_DIR / fold / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line and "def _block_for_fold" in "".join(source):
                new_source.append("    tag = \"emb\" if \"alphaearth\" in fold else \"emb-state\"\n")
                new_source.append("    pred_path = RESULTS_DIR / fold / \"analysis_ageb\" / f\"predictions_oof_{tag}.parquet\"\n")
            elif "metrics_path = RESULTS_DIR / fold / \"analysis_ageb\" / \"metrics_by_state_emb.csv\"" in line and "def _block_for_fold" in "".join(source):
                new_source.append("    metrics_path = RESULTS_DIR / fold / \"analysis_ageb\" / f\"metrics_by_state_{tag}.csv\"\n")
            # 7 & 8. state characteristics and table 8
            elif "metrics_path = RESULTS_DIR / emb_folder / \"analysis_ageb\" / \"metrics_by_state_emb.csv\"" in line:
                new_source.append("        tag = \"emb\" if \"alphaearth\" in emb_folder else \"emb-state\"\n")
                new_source.append("        metrics_path = RESULTS_DIR / emb_folder / \"analysis_ageb\" / f\"metrics_by_state_{tag}.csv\"\n")
            elif "pred_path    = RESULTS_DIR / emb_folder / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line:
                new_source.append("        pred_path    = RESULTS_DIR / emb_folder / \"analysis_ageb\" / f\"predictions_oof_{tag}.parquet\"\n")
            # 9. cross_temporal
            elif "_path_oof2010 = RESULTS_DIR / \"clay2010\" / \"analysis_ageb\" / \"predictions_oof_emb.parquet\"" in line:
                new_source.append("_path_oof2010 = RESULTS_DIR / \"clay2010\" / \"analysis_ageb\" / \"predictions_oof_emb-state.parquet\"\n")
            # 10. table 8
            elif "p = RESULTS_DIR / folder / analysis / \"predictions_oof_emb.parquet\"" in line:
                new_source.append("    tag = \"emb\" if \"alphaearth\" in folder else \"emb-state\"\n")
                new_source.append("    p = RESULTS_DIR / folder / analysis / f\"predictions_oof_{tag}.parquet\"\n")
            else:
                new_source.append(line)
        cell["source"] = new_source

with open("results_analysis/compare_results.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Done editing notebook.")
