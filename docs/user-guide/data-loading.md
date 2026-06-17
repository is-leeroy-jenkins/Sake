# Data Loading

## 🧭 Purpose

This page explains how to load source data into Sake for analysis, statistics, feature engineering, and model training.

Sake can work with tabular data from CSV files, Excel files, Pandas DataFrames, and budget execution datasets such as File A Account Balances.

## 📥 Supported Inputs

| Input Type | Typical Extension | Use |
|---|---:|---|
| CSV | `.csv` | Standard tabular export and lightweight analytical datasets. |
| Excel | `.xlsx`, `.xls` | Account balances, manually curated datasets, and spreadsheet exports. |
| Pandas DataFrame | In-memory object | Notebook, pipeline, and programmatic workflows. |
| File A Account Balances | `.csv`, `.xlsx` | Budget execution analysis by account and reporting period. |

## 🏛️ Budget Execution Dataset Expectations

For File A or similar budget execution data, useful fields commonly include:

| Field Type | Examples |
|---|---|
| Account identifiers | Treasury Account Symbol, main account code, agency, bureau. |
| Reporting dimensions | Fiscal year, period, availability, account classification. |
| Budgetary measures | Budgetary resources, obligations, outlays, recoveries, balances. |
| Descriptive fields | Account name, program, line name, category, subfunction. |

Exact column names may vary by source file. Review the schema after loading.

## 🧪 CSV Example

    import pandas as pd

    df_accounts = pd.read_csv("data/account_balances.csv")

## 🧪 Excel Example

    import pandas as pd

    df_accounts = pd.read_excel("data/account_balances.xlsx", sheet_name=0)

## 🧪 DataFrame Injection Example

    import pandas as pd

    df_accounts = pd.DataFrame(
        {
            "main_account": ["0107", "0110", "0200"],
            "obligations": [1250000, 840000, 2390000],
            "outlays": [930000, 510000, 1710000]
        }
    )

## ✅ Initial Loading Checks

Immediately after loading, review:

    df_accounts.shape
    df_accounts.head()
    df_accounts.columns
    df_accounts.dtypes
    df_accounts.isna().sum()

## 🧹 Basic Normalization

Before analysis, consider standardizing column names:

    df_accounts.columns = (
        df_accounts.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

This reduces errors caused by inconsistent spacing, capitalization, or punctuation.

## 🔢 Numeric Coercion

Budget data often arrives with formatted numbers, commas, parentheses, or text values. Convert numeric columns carefully:

    numeric_columns = ["obligations", "outlays", "budgetary_resources"]

    for column in numeric_columns:
        if column in df_accounts.columns:
            df_accounts[column] = (
                df_accounts[column]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("(", "-", regex=False)
                .str.replace(")", "", regex=False)
            )
            df_accounts[column] = pd.to_numeric(df_accounts[column], errors="coerce")

## 🧾 Target Selection

For supervised learning, identify the target variable before feature preparation.

| Task | Target Example |
|---|---|
| Classification | Execution category, risk class, high/medium/low label. |
| Regression | Obligations, outlays, unobligated balance, execution rate. |

## 🛑 Common Loading Problems

| Problem | Cause | Correction |
|---|---|---|
| Empty DataFrame | Wrong file path or blank sheet. | Confirm path and sheet name. |
| Wrong headers | Header row not first row. | Use `header=` or clean file before loading. |
| Numeric columns read as text | Commas, currency symbols, or mixed values. | Use numeric coercion after load. |
| Duplicate columns | Exported file includes repeated labels. | Rename or drop duplicate fields. |
| Unexpected missing values | Source blanks, merged cells, or invalid parsing. | Inspect missing-value pattern before modeling. |

## ✅ Loading Checklist

Before continuing to data overview:

- The file loads without exceptions.
- The DataFrame has expected rows and columns.
- Column names are standardized.
- Numeric fields are converted.
- Key identifiers are preserved.
- Missing values are measured, not ignored.
- A target column is identified when modeling is planned.
