# Data

## 🧭 Purpose

The `data` module should contain the ingestion, validation, coercion, cleaning, and shaping routines that turn raw Account Balances, CSV, Excel, or Pandas DataFrame inputs into analysis-ready datasets.

This module is the foundation of the Sake workflow. Reliable statistics, feature engineering, model training, and visualization depend on predictable data structure, stable column handling, and clear validation rules.

## 🧱 Workflow Role

| Area                | Responsibility                                                                                           |
|---------------------|----------------------------------------------------------------------------------------------------------|
| File loading        | Read CSV, Excel, and notebook-provided datasets into Pandas DataFrames.                                  |
| Schema review       | Inspect columns, data types, missing values, and candidate target fields.                                |
| Numeric coercion    | Convert financial values, balances, obligations, outlays, and derived fields into usable numeric values. |
| Data validation     | Identify missing columns, duplicate records, malformed values, and impossible ranges.                    |
| Dataset preparation | Return cleaned DataFrames for statistics, feature engineering, modeling, and visualization.              |

## 🔄 Data Flow

```text
Raw File -> Loaded DataFrame -> Validated Schema -> Cleaned DataFrame -> Analysis-Ready Dataset
```

## ✅ Recommended Validation Checks

| Check | Purpose |
|---|---|
| Shape | Confirm row and column counts. |
| Required columns | Verify that required fields are present before analysis. |
| Missing values | Identify fields requiring imputation, filtering, or review. |
| Numeric ranges | Detect malformed financial values, invalid negatives, or extreme balances. |
| Categorical cardinality | Review fields before encoding or grouping. |
| Target availability | Confirm that supervised-learning targets are present before modeling. |

## 🧪 Example

```python
import pandas as pd

df_accounts = pd.read_excel("file_a_account_balances.xlsx")
df_preview = df_accounts.head()
```

## 📚 Source Reference

::: data
