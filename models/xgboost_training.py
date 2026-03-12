import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

print("Loading dataset...")
df = pd.read_csv("../new/new_data.csv")

# Ensure string types for categorical handling
df["cds_pos"] = df["cds_pos"].astype(str)
df["aa_pos"] = df["aa_pos"].astype(str)

X = df[["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]]
y = df["Germline classification"].map({
    "Benign": 0,
    "Pathogenic": 1
})

print("Encoding categorical features...")
X_enc = pd.get_dummies(
    X,
    columns=X.columns,
    prefix=X.columns
)

print("Creating stratified hold-out set...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print("Training XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.9,
    colsample_bytree=0.9,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

print("\n===== HOLD-OUT SET EVALUATION =====")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Error analysis
fn = cm[1][0]
fp = cm[0][1]

print(f"\nFalse Negatives (Pathogenic predicted as Benign): {fn}")
print(f"False Positives (Benign predicted as Pathogenic): {fp}")

# Save artifacts to working_models_new
output_dir = "../working_models_new"
print("\nSaving model artifacts...")
joblib.dump(model, os.path.join(output_dir, "xgb_model.joblib"))
joblib.dump(X_enc.columns.tolist(), os.path.join(output_dir, "xgb_model_columns.joblib"))

print("XGBoost model and feature columns saved successfully.")