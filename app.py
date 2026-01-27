# Fixed app.py

import sys
import os
import re
import webbrowser
from threading import Timer

import pandas as pd
import joblib
from flask import Flask, render_template, request, flash

# -------------------------
# Helper for resource access
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
AA_LETTER_RE = re.compile(r"^[A-Z=*]$")

# -------------------------
# Load trained model
# -------------------------
model = joblib.load(resource_path("rf_model.joblib"))
model_columns = joblib.load(resource_path("model_columns.joblib"))

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
        return False, "CDS position must be 1–9999."
    if not BASE_RE.match(cds_from):
        return False, "CDS 'from' must be A–Z."
    if not BASE_RE.match(cds_to):
        return False, "CDS 'to' must be A–Z."
    if not AA_LETTER_RE.match(aa_from):
        return False, "AA 'from' must be A–Z, =, or *."
    if not POS_RE.match(aa_pos_raw):
        return False, "AA position must be 1–9999."
    if not AA_LETTER_RE.match(aa_to):
        return False, "AA 'to' must be A–Z, =, or *."

    return True, {
        "cds_pos": str(int(cds_pos_raw)),
        "cds_from": cds_from,
        "cds_to": cds_to,
        "aa_from": aa_from,
        "aa_pos": str(int(aa_pos_raw)),
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
            cds_full = f"{result['cds_pos']}{result['cds_from']}>{result['cds_to']}"
            aa_full = f"{result['aa_from']}{result['aa_pos']}{result['aa_to']}"

            try:
                input_df = pd.DataFrame([result])

                input_enc = pd.get_dummies(
                    input_df,
                    columns=input_df.columns,
                    prefix=input_df.columns
                )

                input_enc = input_enc.reindex(
                    columns=model_columns,
                    fill_value=0
                )

                prediction = model.predict(input_enc)[0]
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

    Timer(1, open_browser).start()
    app.run(debug=False, use_reloader=False)
