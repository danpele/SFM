"""
SFM_ch10_scoring
================
Credit Scoring with Logistic Regression

Description:
- Generate synthetic credit scoring dataset
- Fit logistic regression model
- Evaluate with ROC curve and AUC
- Confusion matrix analysis
- Coefficient importance and odds ratios

Statistics of Financial Markets course
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_curve, auc, confusion_matrix,
                              classification_report, roc_auc_score)
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
print("SFM CHAPTER 10: CREDIT SCORING")
print("=" * 70)

np.random.seed(42)

# =============================================================================
# 1. Generate Synthetic Dataset
# =============================================================================
print("\n1. GENERATING SYNTHETIC DATASET")
print("-" * 40)

n = 5000

# Features
income = np.random.lognormal(mean=10.5, sigma=0.5, size=n)
age = np.random.normal(40, 12, n).clip(18, 75)
debt_ratio = np.random.beta(2, 5, n)
credit_history = np.random.randint(0, 30, n)
num_accounts = np.random.poisson(3, n)
employment_years = np.random.exponential(5, n).clip(0, 40)

# Default probability (logistic model)
z = (-3.0
     - 0.00002 * income
     + 0.02 * age
     + 3.0 * debt_ratio
     - 0.1 * credit_history
     + 0.15 * num_accounts
     - 0.08 * employment_years
     + np.random.normal(0, 0.5, n))
prob_default = 1 / (1 + np.exp(-z))
default = (np.random.uniform(0, 1, n) < prob_default).astype(int)

df = pd.DataFrame({
    'income': income,
    'age': age,
    'debt_ratio': debt_ratio,
    'credit_history': credit_history,
    'num_accounts': num_accounts,
    'employment_years': employment_years,
    'default': default
})

print(f"   Total observations: {n}")
print(f"   Default rate: {default.mean():.4f} ({default.sum()} defaults)")
print(f"   Features: {list(df.columns[:-1])}")

# =============================================================================
# 2. Train/Test Split and Scaling
# =============================================================================
print("\n2. PREPARING DATA")
print("-" * 40)

features = ['income', 'age', 'debt_ratio', 'credit_history',
            'num_accounts', 'employment_years']
X = df[features].values
y = df['default'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

print(f"   Training set: {len(X_train)} "
      f"({y_train.mean():.4f} default rate)")
print(f"   Test set:     {len(X_test)} "
      f"({y_test.mean():.4f} default rate)")

# =============================================================================
# 3. Logistic Regression
# =============================================================================
print("\n3. LOGISTIC REGRESSION")
print("-" * 40)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_sc, y_train)

# Predictions
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]

# Coefficients
print(f"\n   Coefficients (standardized):")
print(f"   {'Feature':<20} {'Coeff':>10} {'Odds Ratio':>12}")
print("   " + "-" * 44)
for feat, coef in zip(features, model.coef_[0]):
    print(f"   {feat:<20} {coef:>10.4f} {np.exp(coef):>12.4f}")
print(f"   {'Intercept':<20} {model.intercept_[0]:>10.4f}")

# =============================================================================
# 4. Model Evaluation
# =============================================================================
print("\n4. MODEL EVALUATION")
print("-" * 40)

auc_score = roc_auc_score(y_test, y_prob)
print(f"   AUC-ROC: {auc_score:.4f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\n   Confusion Matrix:")
print(f"   {'':>12} {'Pred No':>10} {'Pred Yes':>10}")
print(f"   {'Actual No':<12} {tn:>10d} {fp:>10d}")
print(f"   {'Actual Yes':<12} {fn:>10d} {tp:>10d}")

accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = (2 * precision * recall / (precision + recall)
      if (precision + recall) > 0 else 0)

print(f"\n   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")

# =============================================================================
# 5. FIGURE: Credit Scoring Analysis (4-panel)
# =============================================================================
print("\n5. CREATING FIGURE")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
axes[0, 0].plot(fpr, tpr, color='#1A3A6E', linewidth=1.5,
                label=f'Logistic Regression (AUC = {auc_score:.3f})')
axes[0, 0].plot([0, 1], [0, 1], 'k--', linewidth=0.8,
                label='Random Classifier')
axes[0, 0].fill_between(fpr, tpr, alpha=0.1, color='#1A3A6E')
axes[0, 0].set_title('ROC Curve', fontweight='bold')
axes[0, 0].set_xlabel('False Positive Rate')
axes[0, 0].set_ylabel('True Positive Rate')
axes[0, 0].legend(loc='lower right', fontsize=8)

# Panel B: Confusion Matrix Heatmap
im = axes[0, 1].imshow(cm, cmap='Blues', aspect='auto')
axes[0, 1].set_xticks([0, 1])
axes[0, 1].set_yticks([0, 1])
axes[0, 1].set_xticklabels(['No Default', 'Default'])
axes[0, 1].set_yticklabels(['No Default', 'Default'])
axes[0, 1].set_xlabel('Predicted')
axes[0, 1].set_ylabel('Actual')
axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
for i in range(2):
    for j in range(2):
        axes[0, 1].text(j, i, f'{cm[i, j]}',
                        ha='center', va='center', fontsize=14,
                        color='white' if cm[i, j] > cm.max() / 2
                        else 'black')
plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

# Panel C: Coefficient Importance (horizontal bar)
coefs = model.coef_[0]
sorted_idx = np.argsort(np.abs(coefs))
axes[1, 0].barh(range(len(features)),
                coefs[sorted_idx],
                color=['#DC3545' if c > 0 else '#2E7D32'
                       for c in coefs[sorted_idx]],
                alpha=0.7)
axes[1, 0].set_yticks(range(len(features)))
axes[1, 0].set_yticklabels([features[i] for i in sorted_idx])
axes[1, 0].axvline(x=0, color='gray', linewidth=0.5)
axes[1, 0].set_title('Feature Coefficients (Standardized)',
                      fontweight='bold')
axes[1, 0].set_xlabel('Coefficient')

# Panel D: Predicted probability distribution
axes[1, 1].hist(y_prob[y_test == 0], bins=50, density=True, alpha=0.5,
                color='#2E7D32', edgecolor='white',
                label='No Default')
axes[1, 1].hist(y_prob[y_test == 1], bins=50, density=True, alpha=0.5,
                color='#DC3545', edgecolor='white',
                label='Default')
axes[1, 1].axvline(x=0.5, color='gray', linestyle='--', linewidth=1,
                    label='Threshold = 0.5')
axes[1, 1].set_title('Predicted Probability Distribution',
                      fontweight='bold')
axes[1, 1].set_xlabel('Predicted P(Default)')
axes[1, 1].set_ylabel('Density')
axes[1, 1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
                  ncol=3, frameon=False)

plt.tight_layout()
save_fig('ch10_scoring')

print("\n" + "=" * 70)
print("CREDIT SCORING ANALYSIS COMPLETE")
print("=" * 70)
print("\nOutput files:")
print("  - ch10_scoring.pdf: 4-panel credit scoring analysis")
