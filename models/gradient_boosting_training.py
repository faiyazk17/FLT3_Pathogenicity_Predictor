import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("Loading dataset...")
df = pd.read_excel("training_dataset.xlsx")

# Ensure string types
df["cds_pos"] = df["cds_pos"].astype(str)
df["aa_pos"] = df["aa_pos"].astype(str)

X = df[["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]]

# Encode labels numerically
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

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Computing sample weights to penalize false negatives...")
# Increase weight of pathogenic samples
sample_weights = np.where(y_train == 1, 2.0, 1.0)  # weight pathogenic 2x

print("Training Gradient Boosting model...")
model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

print("\n===== HOLD-OUT SET EVALUATION =====")
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

fn = cm[1][0]
fp = cm[0][1]

print(f"\nFalse Negatives (Pathogenic predicted as Benign): {fn}")
print(f"False Positives (Benign predicted as Pathogenic): {fp}")

print("\nSaving model artifacts...")
joblib.dump(model, "gb_model.joblib")
joblib.dump(scaler, "gb_scaler.joblib")
joblib.dump(X_enc.columns.tolist(), "gb_model_columns.joblib")

print("Gradient Boosting model, scaler, and feature columns saved successfully.")