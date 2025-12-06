# Fraud Detection in Online Transactions — main.py

## Overview
This repository contains a single script, `main.py`, implementing an end-to-end machine learning pipeline for detecting fraudulent credit-card transactions. The project downloads a labeled dataset from Kaggle, preprocesses the data, handles severe class imbalance, trains multiple classifiers, evaluates them with robust metrics, and saves the best model and scaler. At the end of execution the script also generates and saves visualizations summarizing the results.

## Problem solved
Online payment systems face the challenge of identifying fraudulent transactions among a very large number of legitimate ones. This project demonstrates a practical approach to detecting fraud by:
- Using a public credit-card transactions dataset (Kaggle's `mlg-ulb/creditcardfraud`).
- Handling extreme class imbalance with SMOTE.
- Training and comparing multiple classifiers (Logistic Regression, Random Forest, XGBoost).
- Evaluating models using precision/recall, ROC AUC, PR AUC and confusion matrices.
- Saving the final model and scaler for production use.

## How it works (brief)
1. **Download dataset** from Kaggle via the `kagglehub` API.
2. **Preprocess**: scale the `Time` and `Amount` features.
3. **Train-test split** (stratified).
4. **Balance** training data with SMOTE (oversampling).
5. **Train** three models (Logistic Regression, Random Forest, XGBoost).
6. **Evaluate** each model (classification report, confusion matrix, ROC AUC, PR AUC).
7. **Save** the best model (`fraud_model_xgboost.pkl`) and the scaler (`scaler.pkl`).
8. **Export** result images: `class_distribution.png` and `model_performance.png`.

> Note: the local virtual environment (`fraud_env`) and downloaded dataset are **not** included in this repo.

## Prerequisites
- **Python 3.8+** installed on your machine.
- Internet connection (to download dataset from Kaggle).
- A GitHub account (to push the repo).

---

## Install dependencies (Mac / Windows)

### Recommended (both)
1. Clone the repository (or create it locally and copy `main.py`).
2. Create and activate a Python virtual environment.
3. Install requirements.

Below are the exact commands to run **after cloning** (or after you create the local repo and add the files).

### macOS (Terminal)
```bash
# 1. Ensure Python3 is available
python3 --version

# If python3 is not installed, install via Homebrew:
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# brew install python

# 2. Create & activate a venv (in project root)
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt

# NOTE: if xgboost install fails on some mac systems, run:
# brew install libomp
# pip install xgboost









Windows (PowerShell)
powershell
Copy code
# 1. Ensure Python 3 is available
python --version

# If Python not installed: download & install from https://www.python.org/downloads/

# 2. Create & activate a venv (in project root)
python -m venv venv
venv\Scripts\Activate.ps1
# or if using cmd:
# venv\Scripts\activate.bat

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
Linux (Ubuntu / Debian)
bash
Copy code
sudo apt update
sudo apt install -y python3 python3-venv python3-pip build-essential
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
How to run
After dependencies installed and virtual environment active:

bash
Copy code
python3 main.py
Outputs produced by the script (in the working directory):

class_distribution.png

model_performance.png

fraud_model_xgboost.pkl

scaler.pkl
