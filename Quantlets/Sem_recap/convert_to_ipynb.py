"""Convert SFM_sem_recap_plots.py to .ipynb."""
import json
from pathlib import Path

ROOT = Path(__file__).parent
quantlets = ["SFM_sem_recap_plots"]


def py_to_cells(source):
    lines = source.splitlines()
    cells = []
    i = 0
    # Module docstring
    if lines and lines[0].startswith('"""'):
        j = 1
        while j < len(lines) and '"""' not in lines[j]:
            j += 1
        doc_lines = lines[1:j]
        cells.append({
            "cell_type": "markdown", "metadata": {},
            "source": [f"# {doc_lines[0].strip() if doc_lines else ''}\n\n"] +
                      [l + "\n" for l in doc_lines[1:]]
        })
        i = j + 1

    body = "\n".join(lines[i:]).strip()
    if body:
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
                src_lines = chunk.splitlines()
                src = [l + "\n" for l in src_lines[:-1]] + \
                      ([src_lines[-1]] if src_lines else [])
                cells.append({
                    "cell_type": "code", "metadata": {},
                    "execution_count": None, "outputs": [],
                    "source": src,
                })
    return cells


def build(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py", "mimetype": "text/x-python",
                "name": "python", "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3", "version": "3.12"
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


for q in quantlets:
    py_path = ROOT / q / f"{q}.py"
    ipynb_path = ROOT / q / f"{q}.ipynb"
    src = py_path.read_text(encoding="utf-8")
    cells = py_to_cells(src)
    nb = build(cells)
    ipynb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                          encoding="utf-8")
    print(f"Wrote {ipynb_path}")
