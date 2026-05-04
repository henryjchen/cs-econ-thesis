import json

with open("results_analysis/compare_results.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "def _load_ageb_oof" in source:
            print(repr(source))
