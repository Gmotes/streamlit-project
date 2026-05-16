import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, RandomizedSearchCV
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Load & encode categoricals ─────────────────────────────────────────────────
df = pd.read_csv("data/Churn_Modelling.csv")
df.drop(columns=["RowNumber", "CustomerId", "Surname"], inplace=True)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])          # Female=0, Male=1
df = pd.get_dummies(df, columns=["Geography"], drop_first=True)

# ── Feature engineering ────────────────────────────────────────────────────────
# Ratio of balance to salary — high ratio may signal financial stress / wealth
df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

# Customers with exactly zero balance are a known churn risk segment
df["ZeroBalance"] = (df["Balance"] == 0).astype(int)

# Product density relative to how long they've been a customer
df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

# Active member who also holds a balance — stronger engagement signal
df["ActiveWithBalance"] = df["IsActiveMember"] * (df["Balance"] > 0).astype(int)

# Credit score per age year — proxy for financial trajectory
df["CreditScorePerAge"] = df["CreditScore"] / df["Age"]

# Binned age groups: young (<35), mid (35-55), senior (>55)
df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 35, 55, 100], labels=[0, 1, 2]).astype(int)

X = df.drop(columns=["Exited"])
y = df["Exited"]

engineered = ["BalanceSalaryRatio", "ZeroBalance", "ProductsPerTenure",
              "ActiveWithBalance", "CreditScorePerAge", "AgeGroup"]
print(f"Dataset : {X.shape[0]} rows × {X.shape[1]} features  |  Churn rate: {y.mean():.2%}")
print(f"Engineered features added: {engineered}\n")

# ── Scaling strategy ───────────────────────────────────────────────────────────
# RobustScaler  → outlier-prone continuous: Balance, Age, BalanceSalaryRatio, CreditScorePerAge
# StandardScaler → well-behaved continuous: CreditScore, EstimatedSalary, Tenure, ProductsPerTenure
# Pass-through   → binary / low-cardinality: Gender, HasCrCard, IsActiveMember, NumOfProducts,
#                  Geography dummies, ZeroBalance, ActiveWithBalance, AgeGroup

robust_cols   = ["Balance", "Age", "BalanceSalaryRatio", "CreditScorePerAge"]
standard_cols = ["CreditScore", "EstimatedSalary", "Tenure", "ProductsPerTenure"]
passthrough_cols = [c for c in X.columns if c not in robust_cols + standard_cols]

preprocessor = ColumnTransformer(transformers=[
    ("robust",   RobustScaler(),   robust_cols),
    ("standard", StandardScaler(), standard_cols),
    ("pass",     "passthrough",    passthrough_cols),
])

# ── Models (tree-based use raw features; linear/distance-based get scaled) ─────
def with_prep(clf):
    return Pipeline([("prep", preprocessor), ("clf", clf)])

models = {
    "Logistic Regression": with_prep(LogisticRegression(max_iter=500, random_state=42)),
    "K-Nearest Neighbors": with_prep(KNeighborsClassifier(n_neighbors=7)),
    "SVM (RBF)":           with_prep(SVC(probability=True, random_state=42)),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=20),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42),
}

cv      = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scoring = ["accuracy", "roc_auc", "f1", "precision", "recall"]

# ── Evaluate ───────────────────────────────────────────────────────────────────
results = []
for name, model in models.items():
    print(f"Training: {name} ...", end=" ", flush=True)
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    results.append({
        "Model":     name,
        "Accuracy":  scores["test_accuracy"].mean(),
        "ROC-AUC":   scores["test_roc_auc"].mean(),
        "F1":        scores["test_f1"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall":    scores["test_recall"].mean(),
        "Acc ± std": f"{scores['test_accuracy'].mean():.4f} ± {scores['test_accuracy'].std():.4f}",
    })
    print(f"ROC-AUC={scores['test_roc_auc'].mean():.4f}")

# ── Summary table ──────────────────────────────────────────────────────────────
summary = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 120)

print("\n" + "="*80)
print("10-Fold CV Results — with feature engineering + Robust/Standard scaling")
print("="*80)
print(summary[["Model","Accuracy","ROC-AUC","F1","Precision","Recall","Acc ± std"]].to_string(index=False))

best = summary.iloc[0]
print(f"\n Best model: {best['Model']}  →  ROC-AUC {best['ROC-AUC']:.4f}  |  Accuracy {best['Accuracy']:.4f}  |  F1 {best['F1']:.4f}")

# ── Threshold optimisation ────────────────────────────────────────────────────
print("\n" + "="*80)
print("Threshold Optimisation — OOF probabilities, maximising F1")
print("="*80)

cv_thresh = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
thresh_rows = []

for name, model_obj in models.items():
    print(f"  Calibrating: {name} ...", end=" ", flush=True)
    proba = cross_val_predict(
        model_obj, X, y,
        cv=cv_thresh,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y, proba)
    # precision_recall_curve returns n+1 points; last has no matching threshold
    f1s = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-9)
    best_idx = f1s.argmax()
    opt_thresh = thresholds[best_idx]

    f1_default = f1_score(y, (proba >= 0.35).astype(int))
    f1_optimal = f1s[best_idx]

    thresh_rows.append({
        "Model":         name,
        "Opt Threshold": round(float(opt_thresh), 3),
        "F1 @ 0.50":     round(f1_default, 4),
        "F1 @ optimal":  round(f1_optimal, 4),
        "ΔF1":           round(f1_optimal - f1_default, 4),
        "Precision":     round(float(precisions[best_idx]), 4),
        "Recall":        round(float(recalls[best_idx]), 4),
    })
    print(f"opt_thresh={opt_thresh:.3f}  ΔF1={f1_optimal - f1_default:+.4f}")

thresh_df = pd.DataFrame(thresh_rows).sort_values("F1 @ optimal", ascending=False)
print("\n" + thresh_df.to_string(index=False))
