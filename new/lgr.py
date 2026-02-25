import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("new_data.csv")

#creating label column
df["label"] = df["Germline classification"].map({
    "Benign": 0,
    "Pathogenic": 1
})

df = df.drop(columns=["Germline classification"])

#One hot encoding
X = df.drop("label", axis=1)
y = df["label"]

X_encoded = pd.get_dummies(X, columns=[
    "cds_from", "cds_to",
    "aa_from", "aa_to"
])

# model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42))
])

# CV "accuracy tests"
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

cv_results = cross_validate(pipeline, X_encoded, y, cv=skf, scoring=scoring)

for m in scoring:
    print(m, ":", cv_results["test_" + m].mean())

# optional: final holdout report (nice for writeup)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\nConfusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nReport:\n", classification_report(y_test, y_pred, zero_division=0))

