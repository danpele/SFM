#!/usr/bin/env python3
"""
make_figs.py -- every chart of the SFM exam materials, in the course style (Quantlets/common/sfm_style.py:
transparent background, legend below the plot, no grey), with market data read through
Quantlets/common/sfm_data.py (data/market, last day 18.09.2026).

  exam/2026/fig1_density_v{1,2}.pdf   June 2026 exam: standardised Normal and alpha-stable densities
                                      (the parameters of the exam statement, scipy levy_stable)
  exam/2026/fig2_vr_v{1,2}.pdf        June 2026 exam: VR(q) with the 95% band from V(q)=2(2q-1)(q-1)/(3qT), T=1000
  exam/practice/figs/*_{ro,en}.pdf    practice set: charts from data/market; numbers -> practice/practice_numbers.json
  exam/bank/figs/*.pdf                problem bank (instructor only, git-ignored): every exam/bank/figs_chNN.py
                                      module defines make(out_dir) -> dict; numbers -> bank/bank_numbers.json

Run from anywhere:  python3 exam/make_figs.py  [--only 2026|practice|bank]
"""
import argparse
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, levy_stable

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'Quantlets', 'common'))
import sfm_data as sd      # noqa: E402
import sfm_style as st     # noqa: E402

DIR_2026 = os.path.join(HERE, '2026')


def ro_fmt(dec):
    """Tick formatter with a decimal comma, a space as thousands separator and a true minus sign."""
    return matplotlib.ticker.FuncFormatter(
        lambda v, _: f'{v:,.{dec}f}'.replace(',', ' ').replace('.', ',').replace('-', '\u2212'))


COMMA1 = ro_fmt(1)
DIR_PRACTICE = os.path.join(HERE, 'practice', 'figs')
DIR_BANK = os.path.join(HERE, 'bank', 'figs')


def save(fig, out_dir, name, png=False):
    st.check_no_grey(fig)
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, name + '.pdf'), bbox_inches='tight', transparent=True)
    if png:
        fig.savefig(os.path.join(out_dir, name + '.png'), bbox_inches='tight', transparent=True, dpi=180)
    plt.close(fig)
    print('   saved', os.path.relpath(os.path.join(out_dir, name + '.pdf'), HERE))


# ----------------------------------------------------------------------------------------------- June 2026 exam
def density_fig(alpha, beta, name):
    """Standardised Normal density and a real alpha-stable density (scale chosen so that the stable law is more
    peaked than N(0,1), with fatter tails), as in the exam statement."""
    x = np.linspace(-6, 6, 401)
    normal = norm.pdf(x)
    gamma = levy_stable.pdf(0.0, alpha, beta) / 0.46
    stable = levy_stable.pdf(x, alpha, beta, loc=0, scale=gamma)
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    ax.plot(x, normal, color=st.MainBlue, lw=1.8, label='distribuția Normală $N(0,1)$')
    a_s, b_s = f'{alpha:g}'.replace('.', '{,}'), f'{beta:g}'.replace('.', '{,}')
    ax.plot(x, stable, color=st.IDAred, lw=1.8, label=fr'$\alpha$-stabilă ($\alpha={a_s}$; $\beta={b_s}$)')
    m = x <= -2.0
    ax.fill_between(x[m], normal[m], stable[m], where=stable[m] > normal[m], color=st.IDAred, alpha=0.18, lw=0)
    ax.annotate('coadă stîngă\nmai groasă', xy=(-3.6, levy_stable.pdf(-3.6, alpha, beta, scale=gamma)),
                xytext=(-5.9, 0.22), fontsize=9, color=st.IDAred,
                arrowprops=dict(arrowstyle='->', color=st.IDAred, lw=1))
    ax.set_xlim(-6, 6)
    ax.xaxis.set_major_formatter(ro_fmt(0))
    ax.set_ylim(0, 0.5)
    ax.yaxis.set_major_formatter(COMMA1)
    ax.set_xlabel('randament standardizat')
    ax.set_ylabel('densitate')
    st.legend_outside_bottom(ax, ncol=2, y=-0.24, fontsize=8.5, columnspacing=1.4, handlelength=1.6)
    save(fig, DIR_2026, name)


def vr_fig(qpts, vrpts, ylim, name, T=1000):
    qc = np.linspace(2, 10, 200)
    half = 1.96 * np.sqrt(2.0 * (2 * qc - 1) * (qc - 1) / (3.0 * qc * T))
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.fill_between(qc, 1 - half, 1 + half, color=st.MainBlue, alpha=0.12, lw=0, label='banda 95% sub $H_0$')
    ax.axhline(1.0, color=st.Forest, lw=1.4, ls='--', label='mers aleator: VR = 1')
    ax.plot(qpts, vrpts, 'o-', color=st.IDAred, lw=1.6, ms=6, label='VR(q) estimat')
    ax.set_xticks(qpts)
    ax.set_xlim(2, 10)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(ro_fmt(2))
    ax.set_xlabel('orizont q (zile)')
    ax.set_ylabel('VR(q)')
    st.legend_outside_bottom(ax, ncol=3, y=-0.24, fontsize=8.5, columnspacing=1.1, handlelength=1.6)
    save(fig, DIR_2026, name)


def make_2026():
    print('June 2026 exam')
    density_fig(1.7, -0.3, 'fig1_density_v1')
    density_fig(1.6, -0.2, 'fig1_density_v2')
    vr_fig([2, 5, 10], [1.04, 1.10, 1.15], (0.80, 1.30), 'fig2_vr_v1')   # V1: inside the band (not rejected)
    vr_fig([2, 5, 10], [1.08, 1.20, 1.30], (0.75, 1.40), 'fig2_vr_v2')   # V2: outside the band (rejected)


# ----------------------------------------------------------------------------------------------- practice set
L = {  # chart labels, RO and EN
    'ro': dict(bet='BET (puncte)', level='nivel', dd='drawdown (%)', r='randamente $r_t$',
               absr='randamente absolute $|r_t|$', band=r'$\pm 1{,}96/\sqrt{n}$', lag='lag (zile)', acf='autocorelație',
               hill=r'$\hat\alpha_k$ (Hill), pierderi zilnice Bitcoin', hband=r'$\pm 1{,}96\,\hat\alpha_k/\sqrt{k}$',
               k='k (numărul de pierderi din coadă)'),
    'en': dict(bet='BET (points)', level='level', dd='drawdown (%)', r='returns $r_t$',
               absr='absolute returns $|r_t|$', band=r'$\pm 1.96/\sqrt{n}$', lag='lag (days)', acf='autocorrelation',
               hill=r'$\hat\alpha_k$ (Hill), Bitcoin daily losses', hband=r'$\pm 1.96\,\hat\alpha_k/\sqrt{k}$',
               k='k (number of tail losses)'),
}


def en_fmt(dec):
    return matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.{dec}f}'.replace('-', '−'))


def make_practice():
    print('practice set')
    num = {}
    # data: BET level and drawdown; S&P 500 ACF; Bitcoin Hill estimates
    p = sd.load_close('bet', start='2015-01-01')
    dd = p / p.cummax() - 1
    i = dd.idxmin()
    num['bet_mdd_pct'] = round(100 * dd.min(), 2)
    num['bet_mdd_date'] = str(i.date())
    num['bet_mdd_peak_date'] = str(p.loc[:i].idxmax().date())
    num['bet_peak_level'] = round(float(p.loc[:i].max()), 2)
    num['bet_trough_level'] = round(float(p.loc[i]), 2)
    rec = p.loc[i:][p.loc[i:] >= p.loc[:i].max()]
    num['bet_recovery_date'] = str(rec.index[0].date()) if len(rec) else None

    x = sd.log_returns('sp500', start='2015-01-01').values
    n = len(x)

    def acf(v, h):
        v = v - v.mean()
        return float(np.sum(v[h:] * v[:-h]) / np.sum(v * v))
    lags = np.arange(1, 21)
    a_r = np.array([acf(x, h) for h in lags])
    a_abs = np.array([acf(np.abs(x), h) for h in lags])
    num['sp500_n'] = n
    num['sp500_acf_r_1_5'] = [round(v, 3) for v in a_r[:5]]
    num['sp500_acf_abs_1_5'] = [round(v, 3) for v in a_abs[:5]]
    band = 1.96 / np.sqrt(n)

    rb = sd.log_returns('btc', start='2015-01-01')
    Ls = np.sort(-rb.values)[::-1]
    ks = np.arange(20, 501)
    hill = np.array([1.0 / np.mean(np.log(Ls[:k] / Ls[k])) for k in ks])
    se = hill / np.sqrt(ks)
    num['btc_n'] = len(rb)
    for k in (50, 100, 200, 400):
        num[f'btc_hill_k{k}'] = round(float(1.0 / np.mean(np.log(Ls[:k] / Ls[k]))), 2)

    for lang in ('ro', 'en'):
        T, F = L[lang], (ro_fmt if lang == 'ro' else en_fmt)
        # (1) BET: index level and drawdown, 2015-2026
        fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True,
                                     gridspec_kw=dict(height_ratios=[1.3, 1]))
        a1.plot(p.index, p.values, color=st.IDAred, lw=1.1, label=T['bet'])
        a1.set_ylabel(T['level'])
        a2.fill_between(dd.index, 100 * dd.values, 0, color=st.MainBlue, alpha=0.35, lw=0, label=T['dd'])
        a2.plot(dd.index, 100 * dd.values, color=st.MainBlue, lw=0.8)
        a2.set_ylabel(T['dd'])
        for a in (a1, a2):
            a.yaxis.set_major_formatter(F(0))
        st.fig_legend_bottom(fig, ncol=2, y=0.0)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        save(fig, DIR_PRACTICE, f'practice_bet_drawdown_{lang}')
        # (2) S&P 500: ACF of returns and of absolute returns, lags 1-20
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.bar(lags - 0.2, a_r, width=0.4, color=st.MainBlue, label=T['r'])
        ax.bar(lags + 0.2, a_abs, width=0.4, color=st.IDAred, label=T['absr'])
        ax.axhline(band, color=st.Forest, ls='--', lw=1.1, label=T['band'])
        ax.axhline(-band, color=st.Forest, ls='--', lw=1.1)
        ax.axhline(0, color=st.DarkText, lw=0.6)
        ax.set_xticks(lags)
        ax.set_xlabel(T['lag'])
        ax.set_ylabel(T['acf'])
        ax.yaxis.set_major_formatter(F(1))
        st.legend_outside_bottom(ax, ncol=3, y=-0.25)
        save(fig, DIR_PRACTICE, f'practice_sp500_acf_{lang}')
        # (3) Bitcoin: Hill plot of daily losses, 2015-2026
        fig, ax = plt.subplots(figsize=(7.0, 3.2))
        ax.plot(ks, hill, color=st.Amber, lw=1.4, label=T['hill'])
        ax.fill_between(ks, hill - 1.96 * se, hill + 1.96 * se, color=st.Amber, alpha=0.18, lw=0, label=T['hband'])
        ax.axhline(2, color=st.IDAred, ls='--', lw=1.1, label=r'$\alpha = 2$')
        ax.set_xlabel(T['k'])
        ax.set_ylabel(r'$\hat\alpha_k$')
        ax.set_ylim(1, 4.5)
        ax.yaxis.set_major_formatter(F(1))
        st.legend_outside_bottom(ax, ncol=3, y=-0.25)
        save(fig, DIR_PRACTICE, f'practice_btc_hill_{lang}')

    out = os.path.join(HERE, 'practice', 'practice_numbers.json')
    json.dump(num, open(out, 'w'), indent=1)
    print('   numbers ->', os.path.relpath(out, HERE))
    return num


# ----------------------------------------------------------------------------------------------- problem bank
def make_bank():
    mods = sorted(glob.glob(os.path.join(HERE, 'bank', 'figs_ch*.py')))
    if not mods:
        print('problem bank: no figure modules (exam/bank is instructor-only and may be absent)')
        return {}
    print('problem bank')
    num = {}
    for m in mods:
        name = os.path.splitext(os.path.basename(m))[0]
        spec = importlib.util.spec_from_file_location(name, m)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        num[name] = mod.make(DIR_BANK) or {}
    out = os.path.join(HERE, 'bank', 'bank_numbers.json')
    json.dump(num, open(out, 'w'), indent=1, default=float)
    print('   numbers ->', os.path.relpath(out, HERE))
    return num


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', choices=['2026', 'practice', 'bank'])
    a = ap.parse_args()
    st.apply()
    plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
                         'legend.fontsize': 9})
    if a.only in (None, '2026'):
        make_2026()
    if a.only in (None, 'practice'):
        make_practice()
    if a.only in (None, 'bank'):
        make_bank()
