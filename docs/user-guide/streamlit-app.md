# Streamlit App

## 🧭 Purpose

This page explains how to launch and use the Sake Streamlit application.

The Streamlit app provides a browser-based workflow for loading data, reviewing data quality, running statistical analysis, preparing features, training models, and reviewing outputs.

## ▶️ Launch

From the project root, activate the virtual environment and run:

    streamlit run app.py

Streamlit should open a browser session automatically. If it does not, open the local URL shown in the terminal.

## 🧱 Application Workflow

| Step | User Action | Result |
|---|---|---|
| Load data | Upload or select source data. | DataFrame is created. |
| Review data | Inspect preview, schema, and missing values. | Data quality is understood. |
| Run statistics | Select descriptive or inferential analysis. | Statistical outputs are generated. |
| Engineer features | Prepare numeric and categorical predictors. | Feature matrix is created. |
| Train models | Select classification or regression workflow. | Models are fit and evaluated. |
| Review outputs | Inspect metrics and plots. | Candidate models are compared. |

## 📥 Uploading Data

Use CSV or Excel files with a clear header row. For budget execution data, preserve account identifiers and reporting dimensions.

Recommended source-file properties:

| Property | Recommendation |
|---|---|
| Header row | One row with clear column names. |
| Numeric values | Avoid mixed text and numeric values where possible. |
| Identifiers | Preserve account codes as text when leading zeros matter. |
| Missing values | Use blanks consistently. |
| Duplicates | Remove unintended duplicate rows before upload when known. |

## 🔎 Data Review

After upload, review:

- number of rows
- number of columns
- column names
- data types
- missing values
- duplicate rows
- numeric summaries
- sample records

## 📊 Running Statistics

Use descriptive statistics to profile the dataset before modeling. Use inferential statistics when the analysis requires relationship testing, group comparison, or significance testing.

## 🧮 Model Training

Before training models:

- select the target column
- confirm whether the task is classification or regression
- exclude identifier-only columns from predictors
- handle missing values
- encode categorical fields
- scale numeric features when needed

## 📈 Reviewing Results

Classification outputs may include:

- accuracy
- precision
- recall
- F1 score
- confusion matrix
- ROC curve
- precision-recall curve

Regression outputs may include:

- R²
- mean absolute error
- mean squared error
- root mean squared error
- predicted-versus-actual plot
- residual plot

## 🛑 Stop the App

Return to the terminal and press:

    Ctrl + C

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Correction |
|---|---|---|
| Browser does not open | Streamlit started but browser launch failed. | Open the local URL shown in the terminal. |
| File upload fails | Unsupported file type or corrupted file. | Use CSV or Excel and confirm the file opens locally. |
| Columns are missing | Header row parsed incorrectly. | Clean the source file and reload. |
| Model training fails | Missing target, nonnumeric features, or missing values. | Complete feature engineering first. |
| Charts do not render | Empty result set or failed model step. | Confirm data and model outputs exist. |

## ✅ Streamlit Checklist

- App starts without errors.
- Data loads successfully.
- Preview and schema are correct.
- Target column is selected.
- Feature preparation completes.
- Model training completes.
- Metrics and plots render.
- Results are reviewed before export or reporting.
