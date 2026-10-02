"""Convert SFM_ch4_*.py scripts to Jupyter notebooks (.ipynb)."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
quantlets = [
    "SFM_ch4_acf_sp500",
    "SFM_ch4_vr_profile",
    "SFM_ch4_car_scenarios",
    "SFM_ch4_rolling_hurst",
    "SFM_ch4_active_passive",
    "SFM_ch4_bubbles",
    "SFM_ch4_rw_sim",
    "SFM_ch4_disposition",
    "SFM_ch4_ma_crossover",
    "SFM_ch4_vr_with_ci",
    "SFM_ch4_car_with_ci",
    "SFM_ch4_amh_rolling",
    "SFM_ch4_ma_pre_post",
    "SFM_ch4_cross_asset",
    "SFM_ch4_bvb_case",
    "SFM_ch4_crypto_eff",
    "SFM_ch4_acf_returns_vs_squared",
    "SFM_ch4_apple_events_grid",
    "SFM_ch4_fx_efficiency",
    "SFM_ch4_subperiod_efficiency",
    "SFM_ch4_vr_bootstrap_ci",
    "SFM_ch4_mv_test",
    "SFM_ch4_rw_vs_nonrw",
    "SFM_ch4_full_pipeline",
]


def py_to_cells(source: str):
    """Split Python source into cells by blank lines before top-level
    comments / section markers, producing markdown + code cells."""
    # Extract module docstring as first markdown cell
    cells = []
    lines = source.splitlines(keepends=False)
    i = 0
    # Module docstring
    if lines and lines[0].startswith('"""'):
        j = 1
        while j < len(lines) and '"""' not in lines[j]:
            j += 1
        doc_lines = lines[1:j]
        doc = "\n".join(doc_lines).strip()
        cells.append({
            "cell_type": "markdown", "metadata": {},
            "source": [f"# {doc_lines[0].strip() if doc_lines else ''}\n\n"] +
                      [l + "\n" for l in doc_lines[1:]]
        })
        i = j + 1

    # Remaining as a single code cell (or split by "# ====" headers if any)
    body = "\n".join(lines[i:]).strip()
    if body:
        # Split by separator comments
        chunks = []
        current = []
        for line in body.splitlines():
            if line.startswith("# ====") or line.startswith("# ----"):
                if current:
                    chunks.append("\n".join(current).strip())
                    current = []
            current.append(line)
        if current:
            chunks.append("\n".join(current).strip())
        for chunk in chunks:
            if chunk:
                cells.append({
                    "cell_type": "code", "metadata": {},
                    "execution_count": None, "outputs": [],
                    "source": [l + "\n" for l in chunk.splitlines()[:-1]] +
                              ([chunk.splitlines()[-1]] if chunk.splitlines() else [])
                })
    return cells


def build_notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3", "language": "python", "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for q in quantlets:
    py_path = ROOT / q / f"{q}.py"
    ipynb_path = ROOT / q / f"{q}.ipynb"
    src = py_path.read_text(encoding="utf-8")
    cells = py_to_cells(src)
    nb = build_notebook(cells)
    ipynb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    print(f"Wrote {ipynb_path}")
