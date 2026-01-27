import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_excel("training_dataset.xlsx")

# Ensure string types
df["cds_pos"] = df["cds_pos"].astype(str)
df["aa_pos"] = df["aa_pos"].astype(str)

X = df[["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]]
y = df["Germline classification"]

# One-hot encode
X_enc = pd.get_dummies(
    X,
    columns=X.columns,
    prefix=X.columns
)

# Train / test split
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=7,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# Save model and columns
joblib.dump(model, "rf_model.joblib")
joblib.dump(X_enc.columns.tolist(), "model_columns.joblib")

print("Model and columns saved successfully.")
