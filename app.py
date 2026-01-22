# Working model app.py; broken executable
import sys
import os
from flask import Flask, render_template, request, flash
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import re
import webbrowser
from threading import Timer

# -------------------------
# Helper to access resources
# -------------------------
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# -------------------------
# Flask app
# -------------------------
app = Flask(
    __name__,
    template_folder=resource_path("templates")
)
app.secret_key = "dev"

# -------------------------
# Regex helpers
# -------------------------
POS_RE = re.compile(r"^[1-9][0-9]{0,3}$")  # 1-9999
BASE_RE = re.compile(r"^[A-Z]$")
AA_LETTER_RE = re.compile(r"^[A-Z\=\*]$")

# -------------------------
# Load dataset & train model
# -------------------------
df = pd.read_excel(resource_path('training_dataset.xlsx'))

X = df[["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]]
y = df["Germline classification"]
cols_to_encode = ["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]

for col in ["cds_pos", "aa_pos"]:
    df[col] = df[col].astype(str)

X_enc = pd.get_dummies(df[cols_to_encode], columns=cols_to_encode, prefix=cols_to_encode)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, random_state=42, stratify=y
)

rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=7, random_state=42, class_weight='balanced'
)
rf_clf.fit(X_train, y_train)

# -------------------------
# Form validation
# -------------------------
def validate_and_normalize(form):
    cds_pos_raw = form.get("cds_pos", "").strip()
    cds_from_raw = form.get("cds_from", "").strip()
    cds_to_raw = form.get("cds_to", "").strip()
    aa_from_raw = form.get("aa_from", "").strip()
    aa_pos_raw = form.get("aa_pos", "").strip()
    aa_to_raw = form.get("aa_to", "").strip()

    cds_from = cds_from_raw.upper()
    cds_to = cds_to_raw.upper()
    aa_from = aa_from_raw.upper()
    aa_to = aa_to_raw.upper()

    if not POS_RE.match(cds_pos_raw):
        return False, "CDS position must be 1-9999."
    if not BASE_RE.match(cds_from):
        return False, "CDS 'from' must be A-Z."
    if not BASE_RE.match(cds_to):
        return False, "CDS 'to' must be A-Z."
    if not AA_LETTER_RE.match(aa_from):
        return False, "AA 'from' must be A-Z or = or *."
    if not POS_RE.match(aa_pos_raw):
        return False, "AA position must be 1-9999."
    if not AA_LETTER_RE.match(aa_to):
        return False, "AA 'to' must be A-Z or = or *."

    cds_pos = str(int(cds_pos_raw))
    aa_pos = str(int(aa_pos_raw))

    return True, {
        "cds_pos": cds_pos,
        "cds_from": cds_from,
        "cds_to": cds_to,
        "aa_from": aa_from,
        "aa_pos": aa_pos,
        "aa_to": aa_to
    }

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    cds_full = ""
    aa_full = ""
    form_values = {
        "cds_pos": "", "cds_from": "", "cds_to": "",
        "aa_from": "", "aa_pos": "", "aa_to": ""
    }

    if request.method == "POST":
        ok, result = validate_and_normalize(request.form)
        if not ok:
            flash(result, "error")
        else:
            cds_pos = result["cds_pos"]
            cds_from = result["cds_from"]
            cds_to = result["cds_to"]
            aa_from = result["aa_from"]
            aa_pos = result["aa_pos"]
            aa_to = result["aa_to"]

            cds_full = f"{cds_pos}{cds_from}>{cds_to}"
            aa_full = f"{aa_from}{aa_pos}{aa_to}"

            try:
                input_df = pd.DataFrame([{
                    "cds_pos": str(cds_pos),
                    "cds_from": cds_from,
                    "cds_to": cds_to,
                    "aa_pos": str(aa_pos),
                    "aa_from": aa_from,
                    "aa_to": aa_to
                }])

                input_enc = pd.get_dummies(
                    input_df,
                    columns=["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"],
                    prefix=["cds_pos", "cds_from", "cds_to", "aa_pos", "aa_from", "aa_to"]
                )
                input_enc = input_enc.reindex(columns=X_enc.columns, fill_value=0)
                prediction = rf_clf.predict(input_enc)[0]
            except Exception:
                prediction = "Unknown"

            form_values.update(result)

    return render_template(
        "index.html",
        prediction=prediction,
        cds_full=cds_full,
        aa_full=aa_full,
        form_values=form_values
    )

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    # Open browser 1 second after Flask starts
    Timer(1, open_browser).start()

    app.run(debug=True, use_reloader=False)