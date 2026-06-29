import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use('Agg')  # stops plt.show() from blocking the script

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.impute import SimpleImputer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ======================
# 1. Load Datasets
# ======================
df_small = pd.read_csv("diabetes.csv")    # Pima dataset
df_large = pd.read_csv("diabetes_012_health_indicators_BRFSS2015.csv")   # Large survey dataset

# ======================
# 2. Preprocess Dataset 2 (250k)
# ======================
# Remap Diabetes_012: 0 = no diabetes, 1 & 2 = diabetic
df_large["Outcome"] = df_large["Diabetes_012"].apply(lambda x: 0 if x == 0 else 1)
df_large = df_large.drop(columns=["Diabetes_012"])

# Sample 15k rows
df_large_sampled = df_large.sample(n=15000, random_state=42).reset_index(drop=True)

print("="*60)
print("  OPTION B: Train & Evaluate Each Dataset Separately")
print("="*60)
print(f"Dataset 1 (Pima):       {df_small.shape[0]} rows, {df_small.shape[1]-1} features")
print(f"Dataset 2 (Survey, sampled): {df_large_sampled.shape[0]} rows, {df_large_sampled.shape[1]-1} features")

# ======================
# 3. Define Models
# ======================
def get_models():
    return {
        "Random Forest": RandomForestClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ======================
# 4. Training Function (reusable for both datasets)
# ======================
def train_and_evaluate(X, y, dataset_label, ax_roc, ax_pr):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = get_models()
    results_summary = []
    best_auc = 0
    best_model_name = None

    print(f"\n{'='*60}")
    print(f"  Dataset: {dataset_label}")
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"{'='*60}")

    for name, model in models.items():
        print(f"\n--- {name} ---")

        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        cv_acc = cross_val_score(pipeline, X, y, cv=skf, scoring="accuracy")
        cv_rec = cross_val_score(pipeline, X, y, cv=skf, scoring="recall")
        print(f"CV Accuracy: {cv_acc.mean():.4f} ± {cv_acc.std():.4f}")
        print(f"CV Recall:   {cv_rec.mean():.4f} ± {cv_rec.std():.4f}")

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax_pr.plot(rec, prec, label=name)

        results_summary.append({
            "Model": name,
            "Dataset": dataset_label,
            "Accuracy": round(report["accuracy"], 4),
            "Precision (Diabetic)": round(report["1"]["precision"], 4),
            "Recall (Diabetic)": round(report["1"]["recall"], 4),
            "F1 (Diabetic)": round(report["1"]["f1-score"], 4),
            "AUC-ROC": round(roc_auc, 4),
            "CV Accuracy (mean)": round(cv_acc.mean(), 4),
            "CV Accuracy (std)": round(cv_acc.std(), 4),
            "CV Recall (mean)": round(cv_rec.mean(), 4),
            "CV Recall (std)": round(cv_rec.std(), 4),
        })

        if roc_auc > best_auc:
            best_auc = roc_auc
            best_model_name = name

    return results_summary, best_model_name, X_train, X_test, y_train, y_test, X, y

# ======================
# 5. Run on Dataset 1 (Pima - 768 rows)
# ======================
X1 = df_small.drop("Outcome", axis=1)
y1 = df_small["Outcome"]
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X1[cols_to_fix] = X1[cols_to_fix].replace(0, np.nan)

fig_roc1, ax_roc1 = plt.subplots(figsize=(8, 6))
fig_pr1, ax_pr1 = plt.subplots(figsize=(8, 6))

results1, best1, X1_train, X1_test, y1_train, y1_test, X1_full, y1_full = train_and_evaluate(
    X1, y1, "Dataset 1 - Pima (768 rows)", ax_roc1, ax_pr1
)

# ======================
# 6. Run on Dataset 2 (Survey - sampled 15k)
# ======================
X2 = df_large_sampled.drop("Outcome", axis=1)
y2 = df_large_sampled["Outcome"]

fig_roc2, ax_roc2 = plt.subplots(figsize=(8, 6))
fig_pr2, ax_pr2 = plt.subplots(figsize=(8, 6))

results2, best2, X2_train, X2_test, y2_train, y2_test, X2_full, y2_full = train_and_evaluate(
    X2, y2, "Dataset 2 - Survey (15k sampled)", ax_roc2, ax_pr2
)

# ======================
# 7. Combined Summary Table
# ======================
all_results = pd.DataFrame(results1 + results2)
print("\n" + "="*60)
print("  FULL COMPARISON SUMMARY (Both Datasets)")
print("="*60)
print(all_results.to_string(index=False))
all_results.to_csv("optionB_model_comparison.csv", index=False)

# ======================
# 8. GridSearchCV — Best Model per Dataset
# ======================
def tune_best_model(best_model_name, X_train, y_train, X_test, y_test, label):
    print(f"\n{'='*60}")
    print(f"  Tuning Best Model for {label}: {best_model_name}")
    print(f"{'='*60}")

    models = get_models()

    if best_model_name == "Random Forest":
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5],
            'model__min_samples_leaf': [1, 2]
        }
    elif best_model_name == "XGBoost":
        param_grid = {
            'model__n_estimators': [100, 200],
            'model__max_depth': [3, 5, 7],
            'model__learning_rate': [0.05, 0.1, 0.2]
        }
    elif best_model_name == "Logistic Regression":
        param_grid = {
            'model__C': [0.01, 0.1, 1, 10],
            'model__solver': ['lbfgs', 'liblinear']
        }
    else:
        param_grid = {
            'model__C': [0.1, 1, 10],
            'model__gamma': ['scale', 'auto']
        }

    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('smote', SMOTE(random_state=42)),
        ('model', models[best_model_name])
    ])

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring="recall", n_jobs=-1)
    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    y_pred = grid.best_estimator_.predict(X_test)
    print(f"\nTuned {best_model_name} on {label}:")
    print(classification_report(y_test, y_pred))

tune_best_model(best1, X1_train, y1_train, X1_test, y1_test, "Dataset 1 (Pima)")
tune_best_model(best2, X2_train, y2_train, X2_test, y2_test, "Dataset 2 (Survey)")

# ======================
# 9. Feature Importance — Dataset 1 (RF)
# ======================
rf1 = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
rf1.fit(X1_train, y1_train)
feat_df1 = pd.DataFrame({
    "Feature": X1.columns,
    "Importance": rf1.named_steps['model'].feature_importances_
}).sort_values("Importance", ascending=True)

# ============================================================
# PRACTICAL APPLICATION — Single Patient Diabetes Prediction
# ============================================================

print("\n" + "="*60)
print("   DIABETES PREDICTION DEMO — Best Model (Random Forest)")
print("="*60)

# Sample patient data
sample_patient = pd.DataFrame([{
    'Pregnancies': 3,
    'Glucose': 148,
    'BloodPressure': 72,
    'SkinThickness': 35,
    'Insulin': 120,
    'BMI': 33.6,
    'DiabetesPedigreeFunction': 0.627,
    'Age': 50
}])

# Replace zeros with NaN just like we did for X1
cols_to_fix = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_fix:
    if col in sample_patient.columns:
        sample_patient[col] = sample_patient[col].replace(0, np.nan)

# rf1 is already trained above — just use it directly
prediction = rf1.predict(sample_patient)
probability = rf1.predict_proba(sample_patient)[0][1]

print("\nPatient Information:")
print(f"  Glucose:                  {sample_patient['Glucose'].values[0]}")
print(f"  BMI:                      {sample_patient['BMI'].values[0]}")
print(f"  Age:                      {sample_patient['Age'].values[0]}")
print(f"  Insulin:                  {sample_patient['Insulin'].values[0]}")
print(f"  Blood Pressure:           {sample_patient['BloodPressure'].values[0]}")
print(f"  Skin Thickness:           {sample_patient['SkinThickness'].values[0]}")
print(f"  Pregnancies:              {sample_patient['Pregnancies'].values[0]}")
print(f"  Diabetes Pedigree Func:   {sample_patient['DiabetesPedigreeFunction'].values[0]}")

print("\nPrediction Result:")
if prediction[0] == 1:
    print("  ⚠️  DIABETIC")
else:
    print("  ✅  NON-DIABETIC")

print(f"  Probability of Diabetes:  {probability:.2%}")
print("="*60)

fig_fi1, ax_fi1 = plt.subplots(figsize=(8, 5))
ax_fi1.barh(feat_df1["Feature"], feat_df1["Importance"], color="steelblue")
ax_fi1.set_title("Feature Importance — Dataset 1 (Pima)")
ax_fi1.set_xlabel("Importance Score")
fig_fi1.tight_layout()
fig_fi1.savefig("optionB_feature_importance_pima.png", dpi=150)
plt.show()

# Feature Importance — Dataset 2 (RF)
rf2 = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('smote', SMOTE(random_state=42)),
    ('model', RandomForestClassifier(random_state=42))
])
rf2.fit(X2_train, y2_train)
feat_df2 = pd.DataFrame({
    "Feature": X2.columns,
    "Importance": rf2.named_steps['model'].feature_importances_
}).sort_values("Importance", ascending=True)

fig_fi2, ax_fi2 = plt.subplots(figsize=(10, 7))
ax_fi2.barh(feat_df2["Feature"], feat_df2["Importance"], color="darkorange")
ax_fi2.set_title("Feature Importance — Dataset 2 (Survey)")
ax_fi2.set_xlabel("Importance Score")
fig_fi2.tight_layout()
fig_fi2.savefig("optionB_feature_importance_survey.png", dpi=150)
plt.show()

# ======================
# 10. ROC & PR Curve Plots
# ======================
for ax, fig, title, fname in [
    (ax_roc1, fig_roc1, "Models — ROC Curves (Dataset 1 Pima)", "optionB_roc_pima.png"),
    (ax_roc2, fig_roc2, "Models — ROC Curves (Dataset 2 Survey)", "optionB_roc_survey.png"),
]:
    ax.plot([0, 1], [0, 1], 'k--', label="Random Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.show()

for ax, fig, title, fname in [
    (ax_pr1, fig_pr1, "Models — PR Curves (Dataset 1 Pima)", "optionB_pr_pima.png"),
    (ax_pr2, fig_pr2, "Models — PR Curves (Dataset 2 Survey)", "optionB_pr_survey.png"),
]:
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.show()

# ======================
# 11. Side-by-side AUC Comparison Bar Chart
# ======================
pivot = all_results.pivot(index="Model", columns="Dataset", values="AUC-ROC")
pivot.plot(kind="bar", figsize=(10, 6), colormap="Set2")
plt.title("AUC-ROC Comparison Across Datasets")
plt.ylabel("AUC-ROC Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("optionB_auc_comparison.png", dpi=150)
plt.show()

# ======================
# 12. Confusion Matrix Heatmaps — Both Datasets
# ======================
for ds_label, X_tr, y_tr, X_te, y_te, fname in [
    ("Dataset 1 (Pima)", X1_train, y1_train, X1_test, y1_test, "Diabetes Models_cm_pima.png"),
    ("Dataset 2 (Survey)", X2_train, y2_train, X2_test, y2_test, "Diabetes Models_cm_survey.png"),
]:
    fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for i, (name, model) in enumerate(get_models().items()):
        p = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])
        p.fit(X_tr, y_tr)
        cm = confusion_matrix(y_te, p.predict(X_te))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=["Non-Diabetic", "Diabetic"],
                    yticklabels=["Non-Diabetic", "Diabetic"])
        axes[i].set_title(name)
        axes[i].set_ylabel("Actual")
        axes[i].set_xlabel("Predicted")
    fig_cm.suptitle(f"Diabetes Models — Confusion Matrices ({ds_label})", fontsize=14)
    fig_cm.tight_layout()
    fig_cm.savefig(fname, dpi=150)
    plt.show()

print("\nDiabetes Model complete. All outputs saved with 'Diabetes_model' prefix.")

# # ============================================================
# # PRACTICAL APPLICATION — Single Patient Diabetes Prediction
# # ============================================================

# print("\n" + "="*60)
# print("   DIABETES PREDICTION DEMO — Best Model (Random Forest)")
# print("="*60)

# # Sample patient data — you can change these values
# sample_patient = pd.DataFrame([{
#     'Pregnancies': 3,
#     'Glucose': 148,
#     'BloodPressure': 72,
#     'SkinThickness': 35,
#     'Insulin': 120,
#     'BMI': 33.6,
#     'DiabetesPedigreeFunction': 0.627,
#     'Age': 50
# }])

# # Use rf1 — the Random Forest pipeline already trained on Pima dataset above
# prediction = rf1.predict(sample_patient)
# probability = rf1.predict_proba(sample_patient)[0][1]

# print("\nPatient Information:")
# print(f"  Glucose:                  {sample_patient['Glucose'].values[0]}")
# print(f"  BMI:                      {sample_patient['BMI'].values[0]}")
# print(f"  Age:                      {sample_patient['Age'].values[0]}")
# print(f"  Insulin:                  {sample_patient['Insulin'].values[0]}")
# print(f"  Blood Pressure:           {sample_patient['BloodPressure'].values[0]}")
# print(f"  Skin Thickness:           {sample_patient['SkinThickness'].values[0]}")
# print(f"  Pregnancies:              {sample_patient['Pregnancies'].values[0]}")
# print(f"  Diabetes Pedigree Func:   {sample_patient['DiabetesPedigreeFunction'].values[0]}")

# print("\nPrediction Result:")
# if prediction[0] == 1:
#     print(f"  ⚠️  DIABETIC")
# else:
#     print(f"  ✅  NON-DIABETIC")

# print(f"  Probability of Diabetes:  {probability:.2%}")
# print("="*60)