"""
SFM_ch11_machine_learning
=========================
Machine Learning in Finance: LASSO and Random Forest

Description:
- Download multi-asset data as features
- LASSO regression for return prediction
- Random Forest for return classification
- Feature importance comparison
- Out-of-sample performance evaluation

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, accuracy_score,
                             classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
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
print("SFM CHAPTER 11: MACHINE LEARNING (LASSO & RANDOM FOREST)")
print("=" * 70)

# =============================================================================
# 1. Download Data and Create Features
# =============================================================================
print("\n1. CREATING FEATURE SET")
print("-" * 40)

ticker = 'SPY'
data = yf.download(ticker, start='2010-01-01', end='2024-12-31', progress=False)
close = data['Close'].squeeze()
high = data['High'].squeeze()
low = data['Low'].squeeze()
volume = data['Volume'].squeeze()

# Target: next-day return
returns = np.log(close / close.shift(1))
target = returns.shift(-1)

# Features: lagged returns, volatility, volume, technical indicators
features = pd.DataFrame(index=close.index)
for lag in range(1, 6):
    features[f'ret_lag{lag}'] = returns.shift(lag)

features['vol_5d'] = returns.rolling(5).std()
features['vol_20d'] = returns.rolling(20).std()
features['mom_5d'] = close.pct_change(5)
features['mom_20d'] = close.pct_change(20)
features['rsi'] = 100 - 100 / (1 + (returns.clip(lower=0).rolling(14).mean() /
                                     (-returns.clip(upper=0).rolling(14).mean())))
features['vol_ratio'] = volume / volume.rolling(20).mean()
features['range'] = (high - low) / close
features['ma_ratio'] = close / close.rolling(50).mean()

# Combine
df = features.copy()
df['target'] = target
df.dropna(inplace=True)

feature_names = [c for c in df.columns if c != 'target']
X = df[feature_names].values
y = df['target'].values
y_class = (y > 0).astype(int)

# Time series split
split_idx = int(len(X) * 0.7)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
y_class_train, y_class_test = y_class[:split_idx], y_class[split_idx:]

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"   Features: {len(feature_names)}")
print(f"   Training: {len(X_train)} observations")
print(f"   Testing:  {len(X_test)} observations")

# =============================================================================
# 2. LASSO Regression
# =============================================================================
print("\n2. LASSO REGRESSION")
print("-" * 40)

# Cross-validated LASSO
lasso_cv = LassoCV(alphas=np.logspace(-6, -2, 50), cv=TimeSeriesSplit(5),
                   random_state=42, max_iter=10000)
lasso_cv.fit(X_train_s, y_train)
best_alpha = lasso_cv.alpha_

lasso = Lasso(alpha=best_alpha, max_iter=10000)
lasso.fit(X_train_s, y_train)
lasso_pred = lasso.predict(X_test_s)

rmse_lasso = np.sqrt(mean_squared_error(y_test, lasso_pred))
dir_acc_lasso = np.mean((lasso_pred > 0) == (y_test > 0))

print(f"   Best alpha: {best_alpha:.6f}")
print(f"   RMSE: {rmse_lasso:.6f}")
print(f"   Direction accuracy: {dir_acc_lasso:.4f}")
print(f"   Non-zero coefficients: {np.sum(lasso.coef_ != 0)} / {len(lasso.coef_)}")

print("\n   LASSO Coefficients:")
for name, coef in zip(feature_names, lasso.coef_):
    if coef != 0:
        print(f"     {name:>15}: {coef:>10.6f}")

# =============================================================================
# 3. Random Forest
# =============================================================================
print("\n3. RANDOM FOREST CLASSIFIER")
print("-" * 40)

rf = RandomForestClassifier(n_estimators=200, max_depth=5, min_samples_leaf=20,
                            random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_class_train)
rf_probs = rf.predict_proba(X_test_s)[:, 1]
rf_pred = rf.predict(X_test_s)

acc_rf = accuracy_score(y_class_test, rf_pred)
auc_rf = roc_auc_score(y_class_test, rf_probs)

print(f"   Accuracy:  {acc_rf:.4f}")
print(f"   AUC:       {auc_rf:.4f}")

print("\n   Feature Importances (top 5):")
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i in range(min(5, len(feature_names))):
    idx = sorted_idx[i]
    print(f"     {feature_names[idx]:>15}: {importances[idx]:.4f}")

# =============================================================================
# 4. FIGURE: ML Results (4-panel)
# =============================================================================
print("\n4. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: LASSO coefficient path
alphas_path = np.logspace(-6, -1, 100)
coef_path = []
for a in alphas_path:
    l = Lasso(alpha=a, max_iter=10000)
    l.fit(X_train_s, y_train)
    coef_path.append(l.coef_)
coef_path = np.array(coef_path)

colors_path = plt.cm.tab10(np.linspace(0, 1, len(feature_names)))
for i, (name, c) in enumerate(zip(feature_names, colors_path)):
    axes[0, 0].semilogx(alphas_path, coef_path[:, i], color=c, linewidth=0.8, label=name)
axes[0, 0].axvline(x=best_alpha, color='red', linestyle='--', linewidth=1, label=f'CV α={best_alpha:.2e}')
axes[0, 0].set_title('LASSO Coefficient Path', fontweight='bold')
axes[0, 0].set_xlabel('Regularization (α)')
axes[0, 0].set_ylabel('Coefficient')
axes[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4,
                  frameon=False, fontsize=6)

# Panel B: Feature importance (Random Forest)
sorted_idx_plot = np.argsort(importances)
axes[0, 1].barh([feature_names[i] for i in sorted_idx_plot], importances[sorted_idx_plot],
               color='#1A3A6E', alpha=0.7, edgecolor='white')
axes[0, 1].set_title('Random Forest Feature Importance', fontweight='bold')
axes[0, 1].set_xlabel('Importance')

# Panel C: ROC Curve
fpr, tpr, _ = roc_curve(y_class_test, rf_probs)
lasso_dir_probs = 1 / (1 + np.exp(-lasso_pred * 1000))
fpr_l, tpr_l, _ = roc_curve(y_class_test, lasso_dir_probs)

axes[1, 0].plot(fpr, tpr, color='#1A3A6E', linewidth=1.5, label=f'RF (AUC={auc_rf:.3f})')
axes[1, 0].plot(fpr_l, tpr_l, color='#DC3545', linewidth=1.5,
               label=f'LASSO (AUC={roc_auc_score(y_class_test, lasso_dir_probs):.3f})')
axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=0.8)
axes[1, 0].set_title('ROC Curves', fontweight='bold')
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].legend(loc='lower right', fontsize=8)

# Panel D: Cumulative returns comparison
test_dates = df.index[split_idx:]
cum_rf = np.cumsum(y_test * (2 * (rf_probs > 0.5) - 1))
cum_lasso = np.cumsum(y_test * np.sign(lasso_pred))
cum_bh = np.cumsum(y_test)

axes[1, 1].plot(test_dates, cum_rf, color='#1A3A6E', linewidth=1, label='RF Strategy')
axes[1, 1].plot(test_dates, cum_lasso, color='#DC3545', linewidth=1, label='LASSO Strategy')
axes[1, 1].plot(test_dates, cum_bh, color='#2E7D32', linewidth=1, label='Buy & Hold')
axes[1, 1].set_title('Cumulative Returns: Strategy Comparison', fontweight='bold')
axes[1, 1].set_xlabel('Date')
axes[1, 1].set_ylabel('Cumulative Log Return')
axes[1, 1].legend(loc='upper left', fontsize=8)

plt.tight_layout()
save_fig('ch11_machine_learning')

print("\n" + "=" * 70)
print("MACHINE LEARNING ANALYSIS COMPLETE")
print("=" * 70)
