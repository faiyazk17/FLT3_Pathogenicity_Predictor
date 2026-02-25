import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

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

print("Training Multi-Layer Perceptron (MLP) model...")

# Two hidden layers: 32 and 16 neurons each
model = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)

model.fit(X_train_scaled, y_train)

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
joblib.dump(model, "mlp_model.joblib")
joblib.dump(scaler, "mlp_scaler.joblib")
joblib.dump(X_enc.columns.tolist(), "mlp_model_columns.joblib")

print("MLP model, scaler, and feature columns saved successfully.")