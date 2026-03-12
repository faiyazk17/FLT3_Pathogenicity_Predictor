# FLT3 Pathogenicity Predictor
This repository contains a Machine Learning-based ensemble tool designed to predict the pathogenicity of FLT3 gene variants. By analyzing both Coding DNA Sequence (CDS) and Amino Acid (AA) positions and changes, the tool classifies variants as either Benign or Pathogenic.

The project features a suite of trained models and a Flask-based web interface for easy, real-time inference.

### Project Structure
app.py: The core Flask application that handles the web interface and ensemble logic.

models/: Python scripts used to train individual classifiers (Random Forest, MLP, XGBoost, etc.).

new/: Contains the latest datasets (new_data.csv) used for model refinement.

working_models_new/: Production-ready serialized models (.joblib) and their corresponding feature scalers and column definitions.

templates/: HTML templates for the Flask web UI.

dist/: Contains the standalone executable (FLT3_Predictor.exe) for Windows users.

### How the Ensemble Works
To ensure high reliability, the tool employs a Conservative Voting Ensemble. This logic prioritizes safety in a genomic context:

Input: User provides CDS and AA positions/changes.

Processing: The input is one-hot encoded and scaled to match the training distribution of three distinct models:

Random Forest (High precision)

Multi-Layer Perceptron (Neural Network)

Gradient Boosting (Robustness)

Decision: If any of the three models predicts "Pathogenic," the final output is flagged as Pathogenic. This minimizes False Negatives, ensuring potentially harmful variants are not overlooked.

### Installation & Setup
1. Prerequisites
Python 3.10+ (Tested up to 3.14)

Virtual Environment (recommended)

2. Install Dependencies
Bash
pip install -r requirements.txt
(Ensure scikit-learn, pandas, joblib, xgboost, and Flask are installed.)

3. Running the App
Bash
python app.py
The application will automatically attempt to open http://127.0.0.1:5000 in your default web browser.

### Building the Executable
If you need to distribute the tool as a standalone Windows application:

Ensure you are inside your virtual environment.

Run the PyInstaller command using the provided spec file:

Bash
python -m PyInstaller --clean working_models_new/app.spec
The final executable will be located in the dist/ folder.

### Data & Training
The models were trained on curated FLT3 variant data, utilizing features such as:

cds_pos, cds_from, cds_to

aa_pos, aa_from, aa_to

During training, class imbalance was addressed using balanced class weights and sample weighting to penalize False Negatives heavily, reflecting the clinical importance of identifying pathogenic variants.