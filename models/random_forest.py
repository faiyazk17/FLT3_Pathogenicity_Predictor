import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Configurations

DATA_PATH = "training_dataset.xlsx"
MODEL_PATH = "rf_model.joblib"
COLUMNS_PATH = "model_columns.joblib"

FEATURE_COLS = [
    "cds_pos", "cds_from", "cds_to",
    "aa_pos", "aa_from", "aa_to"
]

LABEL_COL = "Germline classification"

# Load Data

print("Loading dataset...")
df = pd.read_excel(DATA_PATH)

# Ensure positional features are treated as categorical
df["cds_pos"] = df["cds_pos"].astype(str)
df["aa_pos"] = df["aa_pos"].astype(str)

X = df[FEATURE_COLS].copy()
y = df[LABEL_COL]

# One Hot Encoding

print("Encoding categorical features...")
X_enc = pd.get_dummies(
    X,
    columns=FEATURE_COLS,
    prefix=FEATURE_COLS
)

# Train-Test Split

print("Creating stratified hold-out set...")
X_train, X_test, y_train, y_test = train_test_split(
    X_enc,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# Model Training

print("Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Hold-Out Set Evaluation

print("\n===== HOLD-OUT SET EVALUATION =====")

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Extract False Negatives explicitly
# Confusion matrix format:
# [[TN FP]
#  [FN TP]]
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    print(f"\nFalse Negatives (Pathogenic predicted as Benign): {fn}")
    print(f"False Positives (Benign predicted as Pathogenic): {fp}")

# Save Model Artifacts

print("\nSaving model artifacts...")
joblib.dump(model, MODEL_PATH)
joblib.dump(X_enc.columns.tolist(), COLUMNS_PATH)

print("Model and feature columns saved successfully.")