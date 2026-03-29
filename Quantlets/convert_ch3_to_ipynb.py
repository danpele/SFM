"""
Convert all SFM_ch3 Python scripts to Jupyter notebooks + Metainfo.txt
organized in the QuantLet folder structure under Ch_03/.
"""

import json
import os
import re
from datetime import datetime

SRC_DIR = os.path.join(os.path.dirname(__file__), 'SFM_ch3')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'Ch_03')

# Metainfo template
METAINFO_TEMPLATE = """Name of QuantLet: '{name}'

Published in: 'Statistics of Financial Markets (SFM)'

Description: '{description}'

Keywords: '{keywords}'

Author: 'Daniel Traian Pele'

Submitted: '{date}'
"""

# Keywords per quantlet
KEYWORDS = {
    'SFM_ch3_emh_tests': 'efficient market hypothesis, EMH, autocorrelation, Ljung-Box, runs test, ACF, squared returns, volatility clustering, weak-form efficiency',
    'SFM_ch3_variance_ratio': 'variance ratio, Lo-MacKinlay, random walk, rolling variance ratio, weak-form efficiency, heteroscedasticity-robust, S&P 500, Bitcoin',
    'SFM_ch3_hurst_exponent': 'Hurst exponent, R/S analysis, rescaled range, long-range dependence, rolling Hurst, market efficiency, random walk',
    'SFM_ch3_event_study': 'event study, abnormal returns, cumulative abnormal returns, CAR, market model, AAPL earnings, semi-strong efficiency',
    'SFM_ch3_efficiency_comparison': 'market efficiency comparison, cross-asset, autocorrelation, variance ratio, runs test, Hurst exponent, emerging markets, heatmap',
}

# Colab badge URL pattern
COLAB_BADGE = "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danpele/SFM/blob/main/Quantlets/Ch_03/{name}/{name}.ipynb)"


def extract_docstring(code):
    """Extract the triple-quoted docstring from the top of the file."""
    m = re.match(r'^"""(.*?)"""', code, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.match(r"^'''(.*?)'''", code, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ''


def extract_description(docstring):
    """Get the description lines from the docstring."""
    lines = docstring.split('\n')
    desc_lines = []
    started = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('Description:'):
            started = True
            continue
        if started:
            if stripped.startswith('Statistics of Financial Markets'):
                break
            if stripped.startswith('- '):
                desc_lines.append(stripped[2:])
            elif stripped:
                desc_lines.append(stripped)
    return '. '.join(desc_lines) if desc_lines else docstring.split('\n')[0]


def split_sections(code):
    """Split code into sections based on # === ... === markers."""
    code = re.sub(r'^""".*?"""', '', code, count=1, flags=re.DOTALL).strip()
    code = re.sub(r"^'''.*?'''", '', code, count=1, flags=re.DOTALL).strip()

    section_pattern = r'# ={5,}\n# \d+\.\s+(.*?)\n# ={5,}'
    parts = re.split(section_pattern, code)

    sections = []
    if parts[0].strip():
        sections.append(('Setup', parts[0].strip()))

    for i in range(1, len(parts), 2):
        title = parts[i].strip() if i < len(parts) else ''
        body = parts[i + 1].strip() if i + 1 < len(parts) else ''
        if body:
            sections.append((title, body))

    return sections


def adapt_code_for_notebook(code):
    """Adapt code for notebook environment."""
    code = re.sub(
        r"SCRIPT_DIR\s*=\s*os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n",
        '', code)
    code = re.sub(
        r"CHART_DIR\s*=\s*os\.path\.normpath\(os\.path\.join\(SCRIPT_DIR.*?\)\)\n",
        "CHART_DIR = os.path.join('..', '..', '..', 'charts')\n", code)

    # Remove print banners and decorative prints
    code = re.sub(r'print\(["\'][=\-].*?["\']\)\n?', '', code)
    code = re.sub(r'print\("[=\-]" \* \d+\)\n?', '', code)
    code = re.sub(r'print\("\\n" \+ "[=\-]" \* \d+\)\n?', '', code)
    code = re.sub(r'print\(f?"\\n\d+\.\s+.*?"\)\n?', '', code)
    code = re.sub(r'print\(f?"SFM CHAPTER.*?"\)\n?', '', code)
    code = re.sub(r'print\(f?"\\nOutput.*?"\)\n?', '', code)
    code = re.sub(r'print\("Output files:"\)\n?', '', code)
    code = re.sub(r'print\("  - .*?"\)\n?', '', code)
    code = re.sub(r'print\(f?".*?COMPLETE"\)\n?', '', code)
    code = re.sub(r'print\(f?"\d+\.\s+.*?"\)\n?', '', code)

    # Clean up multiple blank lines
    code = re.sub(r'\n{3,}', '\n\n', code)

    return code.strip()


_cell_counter = 0


def make_cell(cell_type, source):
    """Create a notebook cell."""
    global _cell_counter
    lines = source.split('\n')
    source_lines = [line + '\n' for line in lines[:-1]]
    if lines:
        source_lines.append(lines[-1])

    cell_id = f"cell-{_cell_counter:04d}"
    _cell_counter += 1

    cell = {
        "id": cell_id,
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def convert_script(py_path, name):
    """Convert a .py script to a .ipynb notebook."""
    global _cell_counter
    _cell_counter = 0

    with open(py_path, 'r', encoding='utf-8') as f:
        code = f.read()

    docstring = extract_docstring(code)
    description = extract_description(docstring)

    doc_lines = docstring.split('\n')
    title = doc_lines[0] if doc_lines else name

    cells = []

    # Cell 0: Colab badge
    badge = COLAB_BADGE.format(name=name)
    cells.append(make_cell("markdown", badge))

    # Cell 1: Markdown header
    md_header = f"# {title}\n\n"
    for line in doc_lines:
        stripped = line.strip()
        if stripped.startswith('='):
            continue
        if stripped == title:
            continue
        if stripped:
            md_header += stripped + '\n'
    cells.append(make_cell("markdown", md_header.strip()))

    # Split into sections
    sections = split_sections(code)

    for sec_title, sec_code in sections:
        sec_code = adapt_code_for_notebook(sec_code)
        if not sec_code.strip():
            continue

        if sec_title == 'Setup':
            lines = sec_code.split('\n')
            import_lines = []
            settings_lines = []
            in_imports = True

            for line in lines:
                if in_imports and (line.startswith('import ') or
                                   line.startswith('from ') or
                                   line.startswith('warnings.') or
                                   line == '' or
                                   line.startswith('#') and 'Chart style' not in line):
                    import_lines.append(line)
                else:
                    in_imports = False
                    settings_lines.append(line)

            import_block = '%matplotlib inline\n' + '\n'.join(import_lines)
            cells.append(make_cell("code", import_block.strip()))

            if settings_lines:
                settings_block = '\n'.join(settings_lines)
                if settings_block.strip():
                    cells.append(make_cell("code", settings_block.strip()))
        else:
            cells.append(make_cell("markdown", f"## {sec_title}"))
            cells.append(make_cell("code", sec_code))

    if len(cells) <= 2:
        full_code = re.sub(r'^""".*?"""', '', code, count=1, flags=re.DOTALL).strip()
        full_code = adapt_code_for_notebook(full_code)
        cells.append(make_cell("code", '%matplotlib inline\n' + full_code))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat_minor": 0,
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    return notebook, description


def main():
    today = datetime.now().strftime('%A, %d %B %Y')

    py_files = sorted([f for f in os.listdir(SRC_DIR) if f.endswith('.py')])
    print(f"Found {len(py_files)} Python scripts to convert\n")

    for py_file in py_files:
        name = py_file.replace('.py', '')
        py_path = os.path.join(SRC_DIR, py_file)

        # Create output directory
        out_subdir = os.path.join(OUT_DIR, name)
        os.makedirs(out_subdir, exist_ok=True)

        # Convert to notebook
        notebook, description = convert_script(py_path, name)

        # Write notebook
        nb_path = os.path.join(out_subdir, f'{name}.ipynb')
        with open(nb_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        # Write Metainfo.txt
        keywords = KEYWORDS.get(name, 'financial markets, statistics, EMH')
        meta_path = os.path.join(out_subdir, 'Metainfo.txt')
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(METAINFO_TEMPLATE.format(
                name=name,
                description=description,
                keywords=keywords,
                date=today
            ).strip() + '\n')

        print(f"  [OK] {name}/")
        print(f"        -> {name}.ipynb ({len(notebook['cells'])} cells)")
        print(f"        -> Metainfo.txt")

    print(f"\nDone! Output directory: {OUT_DIR}")
    print(f"Total: {len(py_files)} quantlets converted")


if __name__ == '__main__':
    main()
