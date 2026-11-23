"""
Customer Churn Prediction Project
Dataset: Customer-Churn-Records.csv (10,000 rows, target = Exited)

Install requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib

Run:
    python churn_prediction.py
"""

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
DATA_PATH = "Customer-Churn-Records.csv"
TARGET = "Exited"
RANDOM_STATE = 42

# IMPORTANT: In this dataset, 'Complain' is almost identical to 'Exited'
# (customers who complained nearly always left). Using it is DATA LEAKAGE
# and gives ~99.9% accuracy that is useless in real life.
# Keep this False for a realistic model. Set True only to see the leakage.
USE_COMPLAIN = False

# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head(), "\n")
print(df.info(), "\n")
print("Missing values:\n", df.isnull().sum(), "\n")
print("Duplicate rows:", df.duplicated().sum())
print("\nChurn distribution:\n", df[TARGET].value_counts(normalize=True).round(3), "\n")

# ----------------------------------------------------------------------
# 2. EDA
# ----------------------------------------------------------------------
def run_eda(data):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    sns.countplot(x=TARGET, data=data, ax=axes[0, 0])
    axes[0, 0].set_title("Churn Count (0 = Stayed, 1 = Exited)")

    sns.countplot(x="Geography", hue=TARGET, data=data, ax=axes[0, 1])
    axes[0, 1].set_title("Churn by Geography")

    sns.countplot(x="Gender", hue=TARGET, data=data, ax=axes[0, 2])
    axes[0, 2].set_title("Churn by Gender")

    sns.histplot(data=data, x="Age", hue=TARGET, bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title("Age Distribution by Churn")

    sns.countplot(x="NumOfProducts", hue=TARGET, data=data, ax=axes[1, 1])
    axes[1, 1].set_title("Churn by Number of Products")

    sns.countplot(x="IsActiveMember", hue=TARGET, data=data, ax=axes[1, 2])
    axes[1, 2].set_title("Churn by Active Membership")

    plt.tight_layout()
    plt.savefig("eda_plots.png", dpi=120)
    plt.close()

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(12, 9))
    corr = data.select_dtypes(include=np.number).drop(
        columns=["RowNumber", "CustomerId"], errors="ignore"
    ).corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=120)
    plt.close()

    print("Churn rate by Complain:\n", data.groupby("Complain")[TARGET].mean().round(3), "\n")

run_eda(df)

# ----------------------------------------------------------------------
# 3. FEATURE ENGINEERING & CLEANING
# ----------------------------------------------------------------------
# Identifier columns carry no predictive signal
drop_cols = ["RowNumber", "CustomerId", "Surname"]
if not USE_COMPLAIN:
    drop_cols.append("Complain")
df_model = df.drop(columns=drop_cols)

# Rename column with a space for convenience
df_model = df_model.rename(columns={"Satisfaction Score": "SatisfactionScore",
                                    "Card Type": "CardType",
                                    "Point Earned": "PointEarned"})

# Engineered features
df_model["BalanceSalaryRatio"] = df_model["Balance"] / (df_model["EstimatedSalary"] + 1)
df_model["ProductsPerTenure"] = df_model["NumOfProducts"] / (df_model["Tenure"] + 1)
df_model["AgeTenureInteraction"] = df_model["Age"] * df_model["Tenure"]
df_model["HasZeroBalance"] = (df_model["Balance"] == 0).astype(int)

X = df_model.drop(columns=[TARGET])
y = df_model[TARGET]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()
print("Categorical:", categorical_cols)
print("Numeric:", numeric_cols, "\n")

# ----------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols),
    ]
)

# ----------------------------------------------------------------------
# 5. MODELS
# ----------------------------------------------------------------------
# class_weight / scale_pos_weight handle the ~80/20 class imbalance
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        scale_pos_weight=pos_weight, eval_metric="logloss",
        random_state=RANDOM_STATE, n_jobs=-1)

# ----------------------------------------------------------------------
# 6. TRAIN & EVALUATE
# ----------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []
fitted = {}

for name, clf in models.items():
    pipe = Pipeline([("prep", preprocessor), ("model", clf)])
    cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    pipe.fit(X_train, y_train)
    fitted[name] = pipe

    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    results.append({
        "Model": name,
        "CV ROC-AUC": round(cv_auc, 4),
        "Accuracy": round(accuracy_score(y_test, pred), 4),
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "Test ROC-AUC": round(roc_auc_score(y_test, proba), 4),
    })

results_df = pd.DataFrame(results).sort_values("Test ROC-AUC", ascending=False)
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

best_name = results_df.iloc[0]["Model"]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}\n")

# ----------------------------------------------------------------------
# 7. DETAILED EVALUATION OF BEST MODEL
# ----------------------------------------------------------------------
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Stayed", "Exited"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
                       display_labels=["Stayed", "Exited"]).plot(ax=axes[0], cmap="Blues")
axes[0].set_title(f"Confusion Matrix - {best_name}")
for name, pipe in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1], name=name)
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curves")
plt.tight_layout()
plt.savefig("model_evaluation.png", dpi=120)
plt.close()

# ----------------------------------------------------------------------
# 8. THRESHOLD TUNING (optional, useful when recall on churners matters)
# ----------------------------------------------------------------------
best_t, best_f1 = 0.5, 0
for t in np.arange(0.2, 0.8, 0.02):
    f = f1_score(y_test, (y_proba >= t).astype(int))
    if f > best_f1:
        best_t, best_f1 = t, f
print(f"Best threshold for F1: {best_t:.2f} (F1 = {best_f1:.4f})")

# ----------------------------------------------------------------------
# 9. FEATURE IMPORTANCE
# ----------------------------------------------------------------------
model_step = best_model.named_steps["model"]
feature_names = best_model.named_steps["prep"].get_feature_names_out()

if hasattr(model_step, "feature_importances_"):
    importances = model_step.feature_importances_
elif hasattr(model_step, "coef_"):
    importances = np.abs(model_step.coef_[0])
else:
    importances = None

if importances is not None:
    imp = pd.Series(importances, index=feature_names).sort_values(ascending=True).tail(15)
    imp.plot(kind="barh", figsize=(9, 7), title=f"Top Features - {best_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("\nTop 10 features:\n", imp.sort_values(ascending=False).head(10).round(4))

# ----------------------------------------------------------------------
# 10. SAVE MODEL & PREDICT ON NEW CUSTOMER
# ----------------------------------------------------------------------
joblib.dump(best_model, "churn_model.joblib")
print("\nSaved model to churn_model.joblib")

def predict_churn(customer: dict, threshold: float = 0.5):
    """customer: dict with the raw feature columns (see sample below)."""
    model = joblib.load("churn_model.joblib")
    row = pd.DataFrame([customer])
    row["BalanceSalaryRatio"] = row["Balance"] / (row["EstimatedSalary"] + 1)
    row["ProductsPerTenure"] = row["NumOfProducts"] / (row["Tenure"] + 1)
    row["AgeTenureInteraction"] = row["Age"] * row["Tenure"]
    row["HasZeroBalance"] = (row["Balance"] == 0).astype(int)
    row = row[X.columns]  # same column order as training
    prob = model.predict_proba(row)[0, 1]
    return {"churn_probability": round(float(prob), 3), "will_churn": bool(prob >= threshold)}

sample_customer = {
    "CreditScore": 600, "Geography": "Germany", "Gender": "Female", "Age": 45,
    "Tenure": 3, "Balance": 120000.0, "NumOfProducts": 1, "HasCrCard": 1,
    "IsActiveMember": 0, "EstimatedSalary": 70000.0, "SatisfactionScore": 2,
    "CardType": "GOLD", "PointEarned": 400,
}
if USE_COMPLAIN:
    sample_customer["Complain"] = 0

print("\nSample prediction:", predict_churn(sample_customer, threshold=best_t))