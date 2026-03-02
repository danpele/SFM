"""
Convert all SFM_ch1 Python scripts to Jupyter notebooks + Metainfo.txt
organized in the QuantLet folder structure under Ch_01/.
"""

import json
import os
import re
from datetime import datetime

SRC_DIR = os.path.join(os.path.dirname(__file__), 'SFM_ch1')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'Ch_01')

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
    'SFM_ch1_stationarity': 'stationarity, log-returns, price level, S&P 500, I(1), I(0), simple vs log returns',
    'SFM_ch1_returns': 'simple returns, log returns, skewness, kurtosis, Jarque-Bera, MSFT, distribution',
    'SFM_ch1_returns_analysis': 'cumulative returns, density, fat tails, drawdown, maximum drawdown, S&P 500',
    'SFM_ch1_ohlc_orderbook': 'OHLC, candlestick, order book, bid-ask spread, market microstructure, AAPL',
    'SFM_ch1_variance_drag': 'variance drag, geometric return, arithmetic return, volatility, asset classes',
    'SFM_ch1_stylized_facts': 'stylized facts, QQ-plot, ACF, aggregational Gaussianity, fat tails, volatility clustering',
    'SFM_ch1_leverage_effect': 'leverage effect, GJR-GARCH, news impact curve, asymmetric volatility, S&P 500',
    'SFM_ch1_volume_volatility': 'volume, volatility, correlation, scatter plot, S&P 500, volume-volatility',
    'SFM_ch1_volatility_charts': 'rolling volatility, EWMA, efficiency, efficient frontier, CML, tangency portfolio',
    'SFM_ch1_volatility_estimators': 'volatility estimators, Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang, AAPL',
    'SFM_ch1_vol_comparison': 'volatility comparison, conditional, unconditional, GARCH, signature plot, sqrt-T rule',
    'SFM_ch1_sharpe_ratio': 'Sharpe ratio, efficient frontier, portfolio optimization, Monte Carlo, CML, ETF',
}


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
    # Skip name and === underline
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
    # Remove the docstring first
    code = re.sub(r'^""".*?"""', '', code, count=1, flags=re.DOTALL).strip()
    code = re.sub(r"^'''.*?'''", '', code, count=1, flags=re.DOTALL).strip()

    # Split by section markers
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
    # Replace __file__ based paths
    code = re.sub(
        r"SCRIPT_DIR\s*=\s*os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\n",
        '', code)
    code = re.sub(
        r"CHART_DIR\s*=\s*os\.path\.normpath\(os\.path\.join\(SCRIPT_DIR.*?\)\)\n",
        "CHART_DIR = os.path.join('..', '..', '..', 'charts')\n", code)

    # Remove print banners and decorative prints
    # print("=" * 70) and print("-" * 40) style
    code = re.sub(r'print\(["\'][=\-].*?["\']\)\n?', '', code)
    code = re.sub(r'print\("[=\-]" \* \d+\)\n?', '', code)
    # print("\n" + "=" * 70)
    code = re.sub(r'print\("\\n" \+ "[=\-]" \* \d+\)\n?', '', code)
    # print("\n1. DOWNLOADING...")
    code = re.sub(r'print\(f?"\\n\d+\.\s+.*?"\)\n?', '', code)
    # print("SFM CHAPTER...")
    code = re.sub(r'print\(f?"SFM CHAPTER.*?"\)\n?', '', code)
    # print(f"\nOutput directory: ...")
    code = re.sub(r'print\(f?"\\nOutput.*?"\)\n?', '', code)
    code = re.sub(r'print\("Output files:"\)\n?', '', code)
    # print("  - sfm_ch1_xxx.pdf/.png")
    code = re.sub(r'print\("  - .*?"\)\n?', '', code)
    # print("...COMPLETE")
    code = re.sub(r'print\(f?".*?COMPLETE"\)\n?', '', code)
    # Remove section headers: print("\n2. COMPUTING...")  print("-" * 40)
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
        source_lines.append(lines[-1])  # last line without trailing newline

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

    with open(py_path, 'r') as f:
        code = f.read()

    docstring = extract_docstring(code)
    description = extract_description(docstring)

    # Parse docstring for title and description
    doc_lines = docstring.split('\n')
    title = doc_lines[0] if doc_lines else name

    cells = []

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
            # For setup, split imports from chart settings
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

            # Add %matplotlib inline to imports
            import_block = '%matplotlib inline\n' + '\n'.join(import_lines)
            cells.append(make_cell("code", import_block.strip()))

            if settings_lines:
                settings_block = '\n'.join(settings_lines)
                if settings_block.strip():
                    cells.append(make_cell("code", settings_block.strip()))
        else:
            # Add section title as markdown
            cells.append(make_cell("markdown", f"## {sec_title}"))
            # Add the code
            cells.append(make_cell("code", sec_code))

    # If no sections were found, just put everything in one code cell
    if len(cells) <= 1:
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
        with open(nb_path, 'w') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)

        # Write Metainfo.txt
        keywords = KEYWORDS.get(name, 'financial markets, statistics')
        meta_path = os.path.join(out_subdir, 'Metainfo.txt')
        with open(meta_path, 'w') as f:
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
