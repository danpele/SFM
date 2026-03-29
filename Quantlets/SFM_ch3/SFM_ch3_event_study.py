"""
SFM_ch3_event_study
===================
Event Study: AAPL Earnings Announcements

Description:
- Download AAPL and S&P 500 daily data via yfinance
- Identify AAPL earnings dates (fixed for reproducibility)
- Estimate market model on [-250, -11] window
- Compute abnormal returns (AR) and cumulative AR (CAR) on [-5, +20]
- Panel A: Average CAR with 95% confidence band
- Panel B: Average AR bar chart for event window

Statistics of Financial Markets course — Chapter 3 (EMH)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─── Chart style settings — Nature journal quality ───────────────────────────
plt.rcParams['figure.facecolor'] = 'none'
plt.rcParams['axes.facecolor'] = 'none'
plt.rcParams['savefig.facecolor'] = 'none'
plt.rcParams['savefig.transparent'] = True
plt.rcParams['axes.grid'] = False
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.labelsize'] = 9
plt.rcParams['axes.titlesize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75

# ─── Colors ──────────────────────────────────────────────────────────────────
MAIN_BLUE = '#1A3A6E'
CRIMSON   = '#DC3545'
FOREST    = '#2E7D32'
AMBER     = '#B5853F'
ORANGE    = '#E67E22'

# ─── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHART_DIR  = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'charts'))
os.makedirs(CHART_DIR, exist_ok=True)


def save_fig(name):
    """Save figure with transparent background."""
    plt.savefig(os.path.join(CHART_DIR, f'{name}.pdf'),
                bbox_inches='tight', transparent=True)
    plt.savefig(os.path.join(CHART_DIR, f'{name}.png'),
                bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf/.png")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SFM CHAPTER 3: EVENT STUDY (AAPL EARNINGS)")
    print("=" * 70)

    # =========================================================================
    # 1. Download Data
    # =========================================================================
    print("\n1. DOWNLOADING DATA")
    print("-" * 40)

    tickers = ['AAPL', '^GSPC']
    raw = yf.download(tickers, start='2015-01-01', end='2024-12-31',
                      progress=False)['Close']
    if isinstance(raw, pd.Series):
        raw = raw.to_frame()

    # Compute returns
    aapl_ret = raw['AAPL'].dropna().pct_change().dropna()
    sp_ret = raw['^GSPC'].dropna().pct_change().dropna()

    # Align dates
    common_idx = aapl_ret.index.intersection(sp_ret.index)
    aapl_ret = aapl_ret.loc[common_idx]
    sp_ret = sp_ret.loc[common_idx]

    print(f"   AAPL:    {len(aapl_ret)} observations")
    print(f"   S&P 500: {len(sp_ret)} observations")

    # =========================================================================
    # 2. Define Event Dates
    # =========================================================================
    print("\n2. EVENT DATES (AAPL Earnings)")
    print("-" * 40)

    event_dates_str = [
        '2023-01-26', '2023-04-27', '2023-07-27', '2023-10-26',
        '2024-01-25', '2024-05-02', '2024-08-01', '2024-10-31'
    ]

    # Map each event date to the nearest trading day in our index
    event_dates = []
    for d_str in event_dates_str:
        d = pd.Timestamp(d_str)
        # Find nearest trading day at or after the event date
        mask = common_idx >= d
        if mask.any():
            nearest = common_idx[mask][0]
            event_dates.append(nearest)
            print(f"   Event: {d_str} => trading day: {nearest.strftime('%Y-%m-%d')}")

    # =========================================================================
    # 3. Event Study: Market Model
    # =========================================================================
    print("\n3. MARKET MODEL ESTIMATION & ABNORMAL RETURNS")
    print("-" * 40)

    pre_start = -250   # estimation window start (relative to event)
    pre_end   = -11    # estimation window end
    evt_start = -5     # event window start
    evt_end   = 20     # event window end
    event_days = np.arange(evt_start, evt_end + 1)

    all_AR = []
    all_CAR = []
    valid_events = 0

    for event_date in event_dates:
        # Find position of event date in common_idx
        pos = common_idx.get_loc(event_date)

        # Check we have enough data
        if pos + pre_start < 0 or pos + evt_end >= len(common_idx):
            print(f"   Skipping {event_date.strftime('%Y-%m-%d')}: insufficient data")
            continue

        # Estimation window
        est_idx = common_idx[pos + pre_start:pos + pre_end + 1]
        r_stock_est = aapl_ret.loc[est_idx].values
        r_market_est = sp_ret.loc[est_idx].values

        # OLS: r_stock = alpha + beta * r_market + epsilon
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            r_market_est, r_stock_est)

        # Event window
        evt_idx = common_idx[pos + evt_start:pos + evt_end + 1]
        r_stock_evt = aapl_ret.loc[evt_idx].values
        r_market_evt = sp_ret.loc[evt_idx].values

        # Expected returns
        r_expected = intercept + slope * r_market_evt

        # Abnormal returns
        AR = r_stock_evt - r_expected
        CAR = np.cumsum(AR)

        if len(AR) == len(event_days):
            all_AR.append(AR)
            all_CAR.append(CAR)
            valid_events += 1
            print(f"   Event {event_date.strftime('%Y-%m-%d')}: "
                  f"alpha={intercept:.5f}, beta={slope:.3f}, R2={r_value**2:.3f}")

    print(f"\n   Valid events: {valid_events}/{len(event_dates)}")

    # Average across events
    all_AR = np.array(all_AR)    # shape: (n_events, n_days)
    all_CAR = np.array(all_CAR)

    mean_AR = np.mean(all_AR, axis=0)
    mean_CAR = np.mean(all_CAR, axis=0)
    std_CAR = np.std(all_CAR, axis=0, ddof=1)
    n_evt = all_CAR.shape[0]

    # 95% confidence interval
    ci_95 = 1.96 * std_CAR / np.sqrt(n_evt)

    # =========================================================================
    # 4. Print Key Results
    # =========================================================================
    print("\n4. KEY RESULTS")
    print("-" * 40)

    # Find positions for t=0, t=5, t=10
    t0_idx = np.where(event_days == 0)[0][0]
    t5_idx = np.where(event_days == 5)[0][0]
    t10_idx = np.where(event_days == 10)[0][0]

    print(f"   CAR at t=0:   {mean_CAR[t0_idx]*100:>8.4f}%")
    print(f"   CAR at t=+5:  {mean_CAR[t5_idx]*100:>8.4f}%")
    print(f"   CAR at t=+10: {mean_CAR[t10_idx]*100:>8.4f}%")

    # t-test for CAR(0) significance
    t_stat_0 = mean_CAR[t0_idx] / (std_CAR[t0_idx] / np.sqrt(n_evt))
    p_val_0 = 2 * (1 - stats.t.cdf(abs(t_stat_0), df=n_evt - 1))
    print(f"\n   t-test for CAR(0): t={t_stat_0:.3f}, p={p_val_0:.4f}")

    # =========================================================================
    # 5. FIGURE: Event Study
    # =========================================================================
    print("\n5. CREATING FIGURE")
    print("-" * 40)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # --- Panel A: Average CAR with confidence band ---
    ax1.plot(event_days, mean_CAR * 100, color=MAIN_BLUE, linewidth=1.2,
             label='Average CAR')
    ax1.fill_between(event_days,
                     (mean_CAR - ci_95) * 100,
                     (mean_CAR + ci_95) * 100,
                     alpha=0.15, color=MAIN_BLUE, label='95% CI')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.6)
    ax1.axvline(x=0, color=CRIMSON, linestyle='-', linewidth=0.8,
                alpha=0.7, label='Earnings date')

    ax1.set_xlabel('Event day (relative to earnings)')
    ax1.set_ylabel('CAR (%)')
    ax1.set_title('Panel A: Cumulative Abnormal Returns', fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               ncol=3, frameon=False, fontsize=7)

    # Annotate key points
    ax1.annotate(f't=0: {mean_CAR[t0_idx]*100:.2f}%',
                 xy=(0, mean_CAR[t0_idx] * 100),
                 xytext=(3, mean_CAR[t0_idx] * 100 + 0.5),
                 fontsize=7, color=CRIMSON,
                 arrowprops=dict(arrowstyle='->', color=CRIMSON, lw=0.5))

    # --- Panel B: Average AR bar chart ---
    bar_colors = [FOREST if ar >= 0 else CRIMSON for ar in mean_AR]
    ax2.bar(event_days, mean_AR * 100, color=bar_colors,
            alpha=0.7, edgecolor='none', width=0.8)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.6)
    ax2.axvline(x=0, color=CRIMSON, linestyle='-', linewidth=0.8,
                alpha=0.5)

    ax2.set_xlabel('Event day (relative to earnings)')
    ax2.set_ylabel('Average AR (%)')
    ax2.set_title('Panel B: Abnormal Returns by Day', fontweight='bold')

    # Highlight t=0 bar
    t0_color = FOREST if mean_AR[t0_idx] >= 0 else CRIMSON
    ax2.bar(0, mean_AR[t0_idx] * 100, color=t0_color,
            alpha=1.0, edgecolor=MAIN_BLUE, linewidth=0.8, width=0.8)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    save_fig('ch3_event_study')

    # =========================================================================
    # 6. Summary
    # =========================================================================
    print("\n6. SUMMARY")
    print("-" * 40)
    print(f"   Events analyzed: {valid_events} AAPL earnings announcements")
    print(f"   Estimation window: [{pre_start}, {pre_end}]")
    print(f"   Event window: [{evt_start}, +{evt_end}]")
    print("   Market model: r_AAPL = alpha + beta * r_SP500 + epsilon")

    if abs(mean_CAR[t0_idx]) > ci_95[t0_idx]:
        print("   => Significant abnormal return at announcement day")
    else:
        print("   => No significant abnormal return at announcement day")

    post_drift = mean_CAR[t10_idx] - mean_CAR[t0_idx]
    if abs(post_drift) > 0.005:
        print(f"   => Post-earnings drift: {post_drift*100:.3f}% (PEAD)")
    else:
        print("   => No significant post-earnings drift")

    print("\n" + "=" * 70)
    print("EVENT STUDY COMPLETE")
    print("=" * 70)
    print("\nOutput: ch3_event_study.pdf/.png")
