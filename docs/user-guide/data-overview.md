# Data Overview

## 🧭 Purpose

This page explains how to inspect a loaded Sake dataset before running statistics or machine-learning models.

A data overview is a quality-control step. It confirms that the dataset has the expected shape, schema, values, distributions, and identifiers before more expensive analysis begins.

## 📐 Shape and Structure

Start with the dataset dimensions:

    row_count, column_count = df_accounts.shape
    print(row_count, column_count)

Review initial records:

    df_accounts.head()

Review tail records:

    df_accounts.tail()

## 🧾 Column Inventory

Create a column profile:

    df_columns = pd.DataFrame(
        {
            "column": df_accounts.columns,
            "dtype": df_accounts.dtypes.astype(str).values,
            "missing": df_accounts.isna().sum().values,
            "missing_percent": (df_accounts.isna().mean().values * 100).round(2),
            "unique_values": df_accounts.nunique(dropna=True).values
        }
    )

    df_columns

## ✅ Data Quality Review

| Check | Purpose |
|---|---|
| Row count | Confirms expected volume. |
| Column count | Confirms expected schema width. |
| Data types | Separates numeric, categorical, date, and identifier fields. |
| Missing values | Identifies incomplete observations. |
| Unique counts | Flags identifiers, categories, constants, and high-cardinality fields. |
| Duplicates | Detects repeated records. |
| Outliers | Identifies extreme or potentially erroneous values. |

## 🧮 Descriptive Preview

Use summary statistics for numeric fields:

    df_accounts.describe()

Use categorical summaries for object/category fields:

    df_accounts.describe(include=["object", "category"])

## 🔁 Duplicate Review

Check full-row duplicates:

    duplicate_count = df_accounts.duplicated().sum()
    duplicate_count

Check duplicate account-period combinations when relevant:

    key_columns = ["treasury_account_symbol", "fiscal_year", "period"]

    existing_keys = [column for column in key_columns if column in df_accounts.columns]

    if existing_keys:
        df_accounts.duplicated(subset=existing_keys).sum()

## 🧹 Missing-Value Profile

Create a missing-value table:

    df_missing = (
        df_accounts
        .isna()
        .sum()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_count"})
    )

    df_missing["missing_percent"] = (
        df_missing["missing_count"] / len(df_accounts) * 100
    ).round(2)

    df_missing.sort_values("missing_percent", ascending=False)

## 📊 Distribution Review

For numeric budget measures, inspect spread and skew:

    numeric_columns = df_accounts.select_dtypes(include="number").columns
    df_accounts[numeric_columns].describe().T

Look for:

| Pattern | Interpretation |
|---|---|
| Very large max values | A few accounts may dominate totals. |
| Negative values | Recoveries, adjustments, corrections, or data-quality issues. |
| Zero-heavy columns | Sparse activity or inactive accounts. |
| High skew | Model transformations may be needed. |
| Extreme variance | Scaling may be required before some algorithms. |

## 🏛️ Budget Execution Checks

For budget execution data, confirm:

| Check | Question |
|---|---|
| Account identifiers | Are account codes complete and consistently formatted? |
| Fiscal period | Are fiscal years and reporting periods present? |
| Budgetary values | Are obligations, outlays, and balances numeric? |
| Negative values | Are negative amounts valid recoveries or errors? |
| Totals | Do totals reconcile to expected source summaries? |
| Categories | Are classifications or account types complete? |

## 🧪 Example Overview Function

    def summarize_dataframe(df_accounts):
        return {
            "rows": len(df_accounts),
            "columns": len(df_accounts.columns),
            "missing_values": int(df_accounts.isna().sum().sum()),
            "duplicate_rows": int(df_accounts.duplicated().sum()),
            "numeric_columns": list(df_accounts.select_dtypes(include="number").columns),
            "categorical_columns": list(df_accounts.select_dtypes(include=["object", "category"]).columns)
        }

## ✅ Data Overview Checklist

Continue only after confirming:

- The dataset has the expected number of rows.
- Required columns are present.
- Numeric fields have numeric types.
- Missing values are understood.
- Duplicates are reviewed.
- Outliers are identified.
- Account identifiers are preserved.
- A modeling target is selected when needed.
