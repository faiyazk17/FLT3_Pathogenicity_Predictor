import sys
import os
from flask import Flask, render_template, request, flash
import pandas as pd
import joblib
import re
import webbrowser
from threading import Timer

# Helper to access resources (works with PyInstaller)
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

# Flask app
app = Flask(
    __name__,
    template_folder=resource_path("templates")
)
app.secret_key = "dev"

# Regex helpers
POS_RE = re.compile(r"^[1-9][0-9]{0,3}$")  # 1-9999
BASE_RE = re.compile(r"^[A-Z]$")
AA_LETTER_RE = re.compile(r"^[A-Z\=\*]$")

# --- Load NEW models, scalers, and feature columns ---
MODELS_DIR = "working_models_new"

# Random Forest
rf_model = joblib.load(resource_path(os.path.join(MODELS_DIR, "rf_model.joblib")))
rf_columns = joblib.load(resource_path(os.path.join(MODELS_DIR, "model_columns.joblib")))

# MLP
mlp_model = joblib.load(resource_path(os.path.join(MODELS_DIR, "mlp_model.joblib")))
mlp_scaler = joblib.load(resource_path(os.path.join(MODELS_DIR, "mlp_scaler.joblib")))
mlp_columns = joblib.load(resource_path(os.path.join(MODELS_DIR, "mlp_model_columns.joblib")))

# Gradient Boosting
gb_model = joblib.load(resource_path(os.path.join(MODELS_DIR, "gb_model.joblib")))
gb_scaler = joblib.load(resource_path(os.path.join(MODELS_DIR, "gb_scaler.joblib")))
gb_columns = joblib.load(resource_path(os.path.join(MODELS_DIR, "gb_model_columns.joblib")))

# Preprocessing helper
def preprocess_input(df, columns, scaler=None):
    # One-hot encode the input row
    df_enc = pd.get_dummies(df, columns=df.columns, prefix=df.columns)
    # Align with the columns the model was trained on
    df_enc = df_enc.reindex(columns=columns, fill_value=0)
    if scaler:
        df_enc = scaler.transform(df_enc)
    return df_enc

# Ensemble prediction helper
def ensemble_predict(input_df):
    # Preprocess each model input individually using their specific column sets
    rf_input = preprocess_input(input_df.copy(), rf_columns)
    rf_pred = rf_model.predict(rf_input)[0]

    mlp_input = preprocess_input(input_df.copy(), mlp_columns, scaler=mlp_scaler)
    mlp_pred = mlp_model.predict(mlp_input)[0]

    gb_input = preprocess_input(input_df.copy(), gb_columns, scaler=gb_scaler)
    gb_pred = gb_model.predict(gb_input)[0]

    # Map numeric outputs back to labels (RF already uses string labels based on your training output)
    # But for safety with MLP/GB which use 0/1, we check numerically
    
    is_pathogenic = (rf_pred == 1 or rf_pred == "Pathogenic" or 
                    mlp_pred == 1 or 
                    gb_pred == 1)

    final = "Pathogenic" if is_pathogenic else "Benign"

    # Console report for debugging
    print("\n===== NEW PREDICTION =====")
    print(f"CDS: {input_df['cds_pos'].values[0]}{input_df['cds_from'].values[0]}>{input_df['cds_to'].values[0]}")
    print(f"AA: {input_df['aa_from'].values[0]}{input_df['aa_pos'].values[0]}{input_df['aa_to'].values[0]}")
    print(f"Random Forest prediction: {rf_pred}")
    print(f"MLP prediction: {'Pathogenic' if mlp_pred == 1 else 'Benign'}")
    print(f"Gradient Boosting prediction: {'Pathogenic' if gb_pred == 1 else 'Benign'}")
    print(f"Final Ensemble prediction: {final}")

    return final

# Form validation
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

    # Convert to string to ensure matching with training dtypes
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

# Routes
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
                # Prepare input row as a DataFrame
                input_df = pd.DataFrame([{
                    "cds_pos": cds_pos,
                    "cds_from": cds_from,
                    "cds_to": cds_to,
                    "aa_pos": aa_pos,
                    "aa_from": aa_from,
                    "aa_to": aa_to
                }])

                # Ensemble prediction with console report
                prediction = ensemble_predict(input_df)

            except Exception as e:
                prediction = "Unknown"
                print(f"Error during prediction: {e}")

            form_values.update(result)

    return render_template(
        "index.html",
        prediction=prediction,
        cds_full=cds_full,
        aa_full=aa_full,
        form_values=form_values
    )

# Main
if __name__ == "__main__":
    def open_browser():
        webbrowser.open("http://127.0.0.1:5000")

    # Open browser 1 second after Flask starts
    Timer(1, open_browser).start()

    app.run(debug=True, use_reloader=False)