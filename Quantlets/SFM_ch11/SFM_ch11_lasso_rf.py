"""
SFM_ch11_lasso_rf
=================
LASSO and Random Forest for Return Prediction

Description:
- Generate financial features (lagged returns, volatility, momentum)
- LASSO regularization path and cross-validation
- Random Forest feature importance
- Compare prediction performance (R-squared, MSE)
- Demonstrate overfitting vs regularization

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- Standard chart style ---
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
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 0.75
plt.rcParams['legend.facecolor'] = 'none'
plt.rcParams['legend.framealpha'] = 0

def save_fig(name):
    plt.savefig(f'../../charts/{name}.pdf', bbox_inches='tight', transparent=True)
    plt.savefig(f'../../charts/{name}.png', bbox_inches='tight', transparent=True, dpi=300)
    plt.close()
    print(f"   Saved: {name}.pdf")

print("=" * 70)
print("SFM CHAPTER 11: LASSO AND RANDOM FOREST")
print("=" * 70)

# =============================================================================
# 1. Download Data and Create Features
# =============================================================================
print("\n1. DOWNLOADING DATA AND CREATING FEATURES")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31',
                    progress=False)
data.columns = data.columns.get_level_values(0)
close = data['Close']
volume = data['Volume']

# Target: next-day return
log_ret = np.log(close / close.shift(1))

# Feature engineering
features = pd.DataFrame(index=data.index)
for lag in range(1, 11):
    features[f'ret_lag{lag}'] = log_ret.shift(lag)

# Rolling volatility
for w in [5, 10, 20, 60]:
    features[f'vol_{w}d'] = log_ret.rolling(w).std()

# Momentum signals
for w in [5, 10, 20]:
    features[f'mom_{w}d'] = log_ret.rolling(w).sum()

# Volume features
features['vol_ratio'] = volume / volume.rolling(20).mean()
features['vol_change'] = np.log(volume / volume.shift(1))

# RSI approximation
delta = close.diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
features['rsi'] = 100 - 100 / (1 + gain / loss)

# High-Low range
features['range'] = np.log(data['High'] / data['Low'])

# Target
features['target'] = log_ret.shift(-1)

features.dropna(inplace=True)
feature_names = [c for c in features.columns if c != 'target']

print(f"   Ticker: {ticker}")
print(f"   Observations: {len(features)}")
print(f"   Features: {len(feature_names)}")
print(f"   Feature list: {feature_names[:5]} ... ({len(feature_names)} total)")

# =============================================================================
# 2. Train/Test Split
# =============================================================================
print("\n2. PREPARING DATA")
print("-" * 40)

X = features[feature_names].values
y = features['target'].values

# Time-series split: first 70% train, last 30% test
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"   Training: {len(X_train)} obs")
print(f"   Test:     {len(X_test)} obs")

# =============================================================================
# 3. LASSO Regression
# =============================================================================
print("\n3. LASSO REGRESSION")
print("-" * 40)

# Cross-validation for optimal lambda
tscv = TimeSeriesSplit(n_splits=5)
lasso_cv = LassoCV(alphas=np.logspace(-6, -1, 100), cv=tscv,
                    max_iter=10000, random_state=42)
lasso_cv.fit(X_train_sc, y_train)

best_alpha = lasso_cv.alpha_
print(f"   Best alpha (CV): {best_alpha:.6f}")

# Fit final LASSO
lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train_sc, y_train)

y_pred_lasso = lasso.predict(X_test_sc)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

n_nonzero = np.sum(lasso.coef_ != 0)
print(f"   Non-zero coefficients: {n_nonzero}/{len(feature_names)}")
print(f"   Test MSE:  {mse_lasso:.8f}")
print(f"   Test R2:   {r2_lasso:.6f}")

# Regularization path
alphas_path = np.logspace(-6, -1, 100)
coefs_path = []
for a in alphas_path:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_sc, y_train)
    coefs_path.append(l.coef_)
coefs_path = np.array(coefs_path)

# =============================================================================
# 4. Random Forest
# =============================================================================
print("\n4. RANDOM FOREST")
print("-" * 40)

rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                            min_samples_leaf=50, random_state=42,
                            n_jobs=-1)
rf.fit(X_train_sc, y_train)

y_pred_rf = rf.predict(X_test_sc)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"   Test MSE:  {mse_rf:.8f}")
print(f"   Test R2:   {r2_rf:.6f}")

# Feature importance
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
print(f"\n   Top 10 features (RF importance):")
for i in range(min(10, len(feature_names))):
    idx = sorted_idx[i]
    print(f"     {feature_names[idx]:<15}: {importances[idx]:.4f}")

# =============================================================================
# 5. Model Comparison
# =============================================================================
print("\n5. MODEL COMPARISON")
print("-" * 40)

# Naive benchmark: predict mean
y_pred_naive = np.full_like(y_test, y_train.mean())
mse_naive = mean_squared_error(y_test, y_pred_naive)

print(f"   {'Model':<16} {'MSE':>12} {'R2':>10}")
print("   " + "-" * 40)
print(f"   {'Naive (mean)':<16} {mse_naive:>12.8f} {'N/A':>10}")
print(f"   {'LASSO':<16} {mse_lasso:>12.8f} {r2_lasso:>10.6f}")
print(f"   {'Random Forest':<16} {mse_rf:>12.8f} {r2_rf:>10.6f}")

# =============================================================================
# 6. FIGURE: LASSO and RF Analysis (4-panel)
# =============================================================================
print("\n6. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: LASSO Regularization Path
for i in range(coefs_path.shape[1]):
    axes[0, 0].semilogx(alphas_path, coefs_path[:, i],
                         linewidth=0.6, alpha=0.7)
axes[0, 0].axvline(x=best_alpha, color='#DC3545', linestyle='--',
                    linewidth=1, label=f'Best $\\lambda$ = {best_alpha:.5f}')
axes[0, 0].set_title('LASSO Regularization Path', fontweight='bold')
axes[0, 0].set_xlabel('Regularization Parameter ($\\lambda$)')
axes[0, 0].set_ylabel('Coefficient Value')
axes[0, 0].legend(loc='upper right', fontsize=8)

# Panel B: Random Forest Feature Importance (top 15)
top_n = min(15, len(feature_names))
top_idx = sorted_idx[:top_n][::-1]
axes[0, 1].barh(range(top_n),
                importances[top_idx],
                color='#1A3A6E', alpha=0.7)
axes[0, 1].set_yticks(range(top_n))
axes[0, 1].set_yticklabels([feature_names[i] for i in top_idx],
                            fontsize=7)
axes[0, 1].set_title('Random Forest: Feature Importance',
                      fontweight='bold')
axes[0, 1].set_xlabel('Importance')

# Panel C: LASSO non-zero coefficients
nonzero_mask = lasso.coef_ != 0
if np.any(nonzero_mask):
    nz_names = [feature_names[i] for i in range(len(feature_names))
                if nonzero_mask[i]]
    nz_coefs = lasso.coef_[nonzero_mask]
    sort_idx = np.argsort(np.abs(nz_coefs))
    axes[1, 0].barh(range(len(nz_names)),
                    nz_coefs[sort_idx],
                    color=['#DC3545' if c > 0 else '#2E7D32'
                           for c in nz_coefs[sort_idx]],
                    alpha=0.7)
    axes[1, 0].set_yticks(range(len(nz_names)))
    axes[1, 0].set_yticklabels([nz_names[i] for i in sort_idx],
                                fontsize=7)
    axes[1, 0].axvline(x=0, color='gray', linewidth=0.5)
axes[1, 0].set_title('LASSO: Selected Features', fontweight='bold')
axes[1, 0].set_xlabel('Coefficient')

# Panel D: Cumulative prediction accuracy
cum_actual = np.cumsum(y_test)
cum_lasso = np.cumsum(y_pred_lasso)
cum_rf = np.cumsum(y_pred_rf)
test_dates = features.index[split_idx:]

axes[1, 1].plot(test_dates, cum_actual, color='gray', linewidth=0.8,
                label='Actual')
axes[1, 1].plot(test_dates, cum_lasso, color='#1A3A6E', linewidth=0.8,
                label=f'LASSO (R2={r2_lasso:.4f})')
axes[1, 1].plot(test_dates, cum_rf, color='#DC3545', linewidth=0.8,
                label=f'RF (R2={r2_rf:.4f})')
axes[1, 1].set_title('Cumulative Returns: Actual vs Predicted',
                      fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cumulative Log Return')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch11_lasso_rf')

print("\n" + "=" * 70)
print("LASSO AND RANDOM FOREST ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch11_lasso_rf.pdf: 4-panel ML analysis")
