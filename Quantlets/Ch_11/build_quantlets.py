"""Build proper Quantlet folders for Ch_11.

For each quantlet name referenced by slides:
- Ensure folder exists at SFM_ch{N}_{name}/
- Generate Metainfo.txt (if missing) using a template
- Generate .ipynb wrapper that imports fig_{name} from ../generate_all_charts.py
- Copy the chart pdf + png (from D:/G/SFM/charts/) into the folder

Then list all created/updated paths.
"""
import json
import os
import shutil
from pathlib import Path
import sys
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).parent
CHARTS = Path("D:/G/SFM/charts")
REPO_URL = "https://github.com/danpele/SFM"

# Folder name → (fig_function_suffix, chart_basename, human_description)
QUANTLETS = {
    "SFM_ch11_acerbi_szekely": ("acerbi_szekely", "ch11_acerbi_szekely",
        "Acerbi-Szekely Z1/Z2 ES backtesting under H0; Monte Carlo distribution and rejection region."),
    "SFM_ch11_block_maxima": ("block_maxima", "ch11_block_maxima",
        "EVT Block Maxima method; fits GEV to monthly maxima of negative returns."),
    "SFM_ch11_bvb_case": ("bvb_case", "ch11_bvb_case",
        "BET (Bucharest Stock Exchange) case study: VaR rolling estimates and realised exceedances."),
    "SFM_ch11_christoffersen": ("christoffersen_clustering", "ch11_christoffersen_clustering",
        "Christoffersen independence/conditional coverage test; clustered vs independent VaR exceedances."),
    "SFM_ch11_cornish_fisher": ("cornish_fisher", "ch11_cornish_fisher",
        "Cornish-Fisher VaR adjustment for skewness and kurtosis vs parametric Normal."),
    "SFM_ch11_cross_asset": ("cross_asset", "ch11_cross_asset",
        "Cross-asset comparison of VaR 99% across equities, FX, fixed income and crypto."),
    "SFM_ch11_crypto_var": ("crypto_var", "ch11_crypto_var",
        "VaR estimation for crypto returns (BTC/ETH) under heavy-tail regimes (FTX event included)."),
    "SFM_ch11_es_var_ratio": ("es_var_ratio", "ch11_es_var_ratio",
        "Ratio ES/VaR as a function of confidence level under Normal vs Student-t; sanity check ~1.2-1.5."),
    "SFM_ch11_evt_var": ("evt_var", "ch11_evt_var",
        "Extreme Value Theory POT estimator; fits GPD to peaks over threshold, computes EVT-VaR/ES."),
    "SFM_ch11_filtered_hs": ("filtered_hs", "ch11_filtered_hs",
        "Filtered Historical Simulation (Hull-White 1998): HS on standardised GARCH innovations."),
    "SFM_ch11_garch_var": ("garch_var", "ch11_garch_var",
        "Conditional VaR via GARCH(1,1)-t vs Historical Simulation; reaction to volatility shocks."),
    "SFM_ch11_gjr_garch": ("gjr_garch", "ch11_gjr_garch",
        "GJR-GARCH asymmetric volatility model; leverage effect on conditional VaR."),
    "SFM_ch11_hill_plot": ("hill_plot", "ch11_hill_plot",
        "Hill plot for tail index estimation; stability over choice of k (top order statistics)."),
    "SFM_ch11_kupiec": ("kupiec_distribution", "ch11_kupiec_distribution",
        "Kupiec POF (Proportion of Failures) likelihood-ratio test distribution under H0; chi-square asymptotic."),
    "SFM_ch11_lehman_2008": ("lehman_2008", "ch11_lehman_2008",
        "Lehman 2008 case study: VaR violations during the September 2008 collapse."),
    "SFM_ch11_lopez_loss": ("lopez_loss", "ch11_lopez_loss",
        "Lopez loss function for VaR backtesting; penalises both number and magnitude of breaches."),
    "SFM_ch11_lvar": ("lvar", "ch11_lvar",
        "Liquidity-adjusted VaR (LVaR); bid-ask spread adjustment to classical VaR."),
    "SFM_ch11_method_hs": ("method_hs", "ch11_method_hs",
        "Historical Simulation: empirical quantile of recent returns; transparent and assumption-free."),
    "SFM_ch11_method_mc": ("method_mc", "ch11_method_mc",
        "Monte Carlo VaR and ES under geometric Brownian motion; M=50000 paths, 10-day horizon."),
    "SFM_ch11_method_normal": ("method_normal", "ch11_method_normal",
        "Parametric Normal VaR; closed form -V(mu + sigma z_alpha); benchmark method."),
    "SFM_ch11_method_student_t": ("method_student_t", "ch11_method_student_t",
        "Parametric Student-t VaR vs Normal; heavier tails capture fat-tail risk; ratio rises for small alpha."),
    "SFM_ch11_methods_comparison": ("methods_comparison", "ch11_methods_comparison",
        "Comparative panel: HS, Normal, Student-t, Monte Carlo, GARCH-t and EVT-POT on the same series."),
    "SFM_ch11_rolling_var_es": ("rolling_var_es", "ch11_rolling_var_es",
        "Rolling 250-day VaR and ES at 5% with realised exceedances over a long simulated GARCH-t series."),
    "SFM_ch11_subadditivity": ("subadditivity", "ch11_subadditivity",
        "Counter-example: VaR fails subadditivity (two defaultable bonds); ES is coherent and satisfies it."),
    "SFM_ch11_traffic_light": ("traffic_light", "ch11_traffic_light",
        "Basel traffic-light test: green/yellow/red zones based on number of VaR exceptions in 250 days."),
    "SFM_ch11_var_concept": ("var_concept", "ch11_var_concept",
        "Visual introduction to VaR as a lower tail quantile of the loss distribution."),
    "SFM_ch11_var_horizons": ("var_horizons", "ch11_var_horizons",
        "Scaling VaR across horizons: square-root-of-time rule and its failure under heavy tails / autocorrelation."),
    "SFM_ch11_violation_dist": ("violation_dist", "ch11_violation_dist",
        "Binomial distribution of number of VaR exceptions over 250 days; 95% CI bands at three alpha levels."),
}

METAINFO_TMPL = """Name of QuantLet: '{name}'

Published in: 'Statistics of Financial Markets (SFM)'

Description: '{description}'

Keywords: '{keywords}'

Author: 'Daniel Traian Pele, Antoaneta Amza'

Submitted: 'Tuesday, 12 May 2026'

Datafile: 'simulated GARCH(1,1)-t returns or curated example'

Output: '{chart}.pdf'
"""

def make_keywords(name, description):
    # Pull simple keywords from name plus a few common ones.
    parts = name.replace("SFM_ch11_", "").split("_")
    parts = [p for p in parts if p not in {"and", "of"}]
    return ", ".join(parts + ["VaR", "Expected Shortfall", "backtesting"])

def make_notebook(name, fn_suffix, chart):
    badge = (
        f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
        f"(https://colab.research.google.com/github/danpele/SFM/blob/main/"
        f"Quantlets/Ch_11/{name}/{name}.ipynb)"
    )
    title = f"# {name}\n"
    chart_md = (
        f"![{chart}](./{chart}.png)\n\n"
        f"Output: `{chart}.pdf`"
    )
    code = (
        f"import sys, os\n"
        f"sys.path.insert(0, os.path.abspath('..'))\n"
        f"from generate_all_charts import fig_{fn_suffix}\n\n"
        f"fig_{fn_suffix}()\n"
    )
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": [badge]},
        {"cell_type": "markdown", "metadata": {}, "source": [title]},
        {"cell_type": "code", "metadata": {}, "source": [code],
         "execution_count": None, "outputs": []},
        {"cell_type": "markdown", "metadata": {}, "source": [chart_md]},
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def main():
    created, updated, missing_charts = [], [], []
    for name, (fn_suffix, chart_base, desc) in QUANTLETS.items():
        folder = ROOT / name
        folder.mkdir(exist_ok=True)
        # Metainfo
        meta = folder / "Metainfo.txt"
        if not meta.exists():
            meta.write_text(METAINFO_TMPL.format(
                name=name, description=desc,
                keywords=make_keywords(name, desc), chart=chart_base
            ), encoding="utf-8")
            created.append(str(meta))
        # Notebook
        ipynb = folder / f"{name}.ipynb"
        nb = make_notebook(name, fn_suffix, chart_base)
        ipynb.write_text(json.dumps(nb, indent=1), encoding="utf-8")
        updated.append(str(ipynb))
        # Charts
        for ext in ("pdf", "png"):
            src = CHARTS / f"{chart_base}.{ext}"
            dst = folder / f"{chart_base}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)
            else:
                missing_charts.append(str(src))
        # Remove the old .py wrapper if present (replaced by notebook)
        old_py = folder / f"{name}.py"
        if old_py.exists():
            old_py.unlink()

    print(f"Quantlets processed: {len(QUANTLETS)}")
    print(f"Metainfo created: {len(created)}")
    print(f"Notebooks written: {len(updated)}")
    if missing_charts:
        print(f"WARNING — missing charts ({len(missing_charts)}):")
        for m in missing_charts:
            print(f"  - {m}")

if __name__ == "__main__":
    main()
