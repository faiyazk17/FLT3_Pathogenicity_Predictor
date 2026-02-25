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


#hyperparameter tuning 
from sklearn.model_selection import GridSearchCV

param_grid = {
    "model__C": [0.01, 0.1, 1, 5, 10, 50]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

grid.fit(X_encoded, y)

print("Best C:", grid.best_params_)
print("Best ROC-AUC:", grid.best_score_)

#model with best C
best_model = grid.best_estimator_

results = cross_validate(
    best_model,
    X_encoded, y,
    cv=skf,
    scoring=["accuracy","precision","recall","f1","roc_auc"]
)

for m in ["accuracy","precision","recall","f1","roc_auc"]:
    print(m, ":", results["test_"+m].mean())
    
#interpreting the model with the best c
import numpy as np

best_model.fit(X_encoded, y)
coefs = best_model.named_steps["model"].coef_[0]
feat_names = X_encoded.columns

coef_df = pd.DataFrame({"feature": feat_names, "coef": coefs})
coef_df["abs"] = coef_df["coef"].abs()

print("Top features increasing pathogenic risk:")
print(coef_df.sort_values("coef", ascending=False).head(15)[["feature","coef"]])

print("\nTop features decreasing pathogenic risk (more benign):")
print(coef_df.sort_values("coef", ascending=True).head(15)[["feature","coef"]])
