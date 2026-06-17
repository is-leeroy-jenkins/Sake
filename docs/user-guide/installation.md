# Installation

## 🧭 Purpose

This page explains how to prepare a local Sake environment using Python, a virtual environment, and the project dependency file.

The recommended local workflow uses a project-specific virtual environment so dependencies remain isolated from other Python projects.

## ✅ Prerequisites

| Requirement | Recommendation |
|---|---|
| Python | Python 3.9 through 3.12, 64-bit preferred. |
| Terminal | PowerShell in PyCharm, Windows Terminal, or a standard shell. |
| Package manager | `pip` installed with Python. |
| Repository | Sake source code cloned or downloaded locally. |

## 📥 Clone the Repository

Run this from the directory where you keep source repositories:

    git clone https://github.com/is-leeroy-jenkins/Sake.git
    cd Sake

When using a fork, replace the repository URL with the fork URL.

## 🐍 Create a Virtual Environment

### Windows PowerShell

    python -m venv .venv
    .venv\Scripts\Activate.ps1

### macOS or Linux

    python3 -m venv .venv
    source .venv/bin/activate

After activation, the terminal prompt should show `.venv`.

## 📦 Install Dependencies

Upgrade packaging tools first:

    python -m pip install --upgrade pip wheel setuptools

Install project requirements:

    pip install -r requirements.txt

## 🧪 Confirm the Environment

Run these checks from the project root:

    python --version
    python -m pip --version
    python -c "import pandas, sklearn, streamlit; print('environment ok')"

## ▶️ Launch the Streamlit App

    streamlit run app.py

When Streamlit starts, it normally opens a browser automatically. If it does not, use the local URL shown in the terminal, commonly:

    http://localhost:8501

## 📓 Notebook Runtime

For notebook usage, install Jupyter if it is not already available:

    pip install jupyter
    jupyter notebook

Open the Sake notebook and run the cells from top to bottom.

## ☁️ Cloud Runtime Notes

Sake can also be used in hosted notebook environments such as Google Colab or Databricks. In those environments, confirm that all project dependencies are installed in the active runtime before executing notebook cells.

## 🛠️ Common Installation Issues

| Symptom | Likely Cause | Correction |
|---|---|---|
| `streamlit` is not recognized | Dependency not installed or virtual environment inactive. | Activate `.venv` and run `pip install -r requirements.txt`. |
| Import error for `sklearn` | Scikit-learn missing from the environment. | Run `pip install scikit-learn`. |
| Excel file read error | Excel reader missing. | Confirm `openpyxl` and `xlrd` are installed. |
| Wrong Python version | Shell uses another Python installation. | Run `where python` on Windows or `which python` on macOS/Linux. |
| Execution policy blocks activation | PowerShell policy prevents scripts. | Use an approved execution policy change or activate through PyCharm’s configured interpreter. |

## ✅ Validation Checklist

Before moving to data loading, confirm:

- The virtual environment is active.
- `pip install -r requirements.txt` completed successfully.
- `python -c "import pandas, sklearn, streamlit"` runs without errors.
- `streamlit run app.py` starts the application.
- The project root contains `app.py`, `requirements.txt`, and the documentation files.
