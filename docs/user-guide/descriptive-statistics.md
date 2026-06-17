# Descriptive Statistics

## 🧭 Purpose

This page explains the descriptive statistics used in Sake to summarize budget execution and tabular analytical datasets.

Descriptive statistics establish the baseline condition of the data before inference, feature engineering, or machine learning. They answer traditional questions: how large, how common, how variable, how skewed, and how extreme.

## 📊 Core Statistics

| Statistic | Description | Budget Execution Use |
|---|---|---|
| Mean | Arithmetic average. | Average obligations, outlays, or balances across accounts. |
| Median | Middle value after sorting. | Robust central tendency for skewed financial data. |
| Mode | Most frequent value. | Common classifications, account codes, or availability values. |
| Standard deviation | Average spread around the mean. | Variability in execution values. |
| Variance | Squared spread around the mean. | Model diagnostics and dispersion analysis. |
| Minimum | Smallest observed value. | Negative adjustments or low-activity accounts. |
| Maximum | Largest observed value. | Dominant accounts or extreme outlays. |
| Range | Maximum minus minimum. | Total spread of a field. |
| Interquartile range | Spread between 25th and 75th percentiles. | Robust outlier review. |
| Skewness | Distribution asymmetry. | Identifies accounts where a few values dominate. |
| Kurtosis | Tail weight and peakedness. | Flags outlier-prone distributions. |

## 🧪 Numeric Summary

    numeric_columns = df_accounts.select_dtypes(include="number").columns

    df_summary = df_accounts[numeric_columns].describe().T
    df_summary

## 🧪 Extended Summary

    df_extended = pd.DataFrame(
        {
            "mean": df_accounts[numeric_columns].mean(),
            "median": df_accounts[numeric_columns].median(),
            "std": df_accounts[numeric_columns].std(),
            "variance": df_accounts[numeric_columns].var(),
            "skewness": df_accounts[numeric_columns].skew(),
            "kurtosis": df_accounts[numeric_columns].kurtosis(),
            "missing": df_accounts[numeric_columns].isna().sum()
        }
    )

    df_extended

## 🧮 Percentile Review

Percentiles help describe spread without relying only on averages.

    df_percentiles = df_accounts[numeric_columns].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    df_percentiles

## 📌 Interquartile Range

Use IQR to identify potential outliers:

    q1 = df_accounts[numeric_columns].quantile(0.25)
    q3 = df_accounts[numeric_columns].quantile(0.75)
    iqr = q3 - q1

    df_iqr = pd.DataFrame(
        {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": q1 - 1.5 * iqr,
            "upper_fence": q3 + 1.5 * iqr
        }
    )

    df_iqr

## 🏛️ Budget Execution Interpretation

| Pattern | Possible Interpretation |
|---|---|
| Mean much larger than median | A few high-value accounts dominate totals. |
| Large standard deviation | Wide variation across accounts or periods. |
| High positive skew | Most accounts are small while a few are very large. |
| Negative minimum | Recoveries, corrections, cancellations, or data issues. |
| Zero median | Many inactive, no-year, or sparse accounts. |
| High kurtosis | Extreme values require closer review. |

## 📊 Grouped Statistics

For account, agency, or category comparisons:

    df_grouped = (
        df_accounts
        .groupby("account_type")
        .agg(
            row_count=("account_type", "size"),
            mean_obligations=("obligations", "mean"),
            median_obligations=("obligations", "median"),
            total_obligations=("obligations", "sum")
        )
        .reset_index()
    )

    df_grouped

Replace `account_type` and `obligations` with columns from the active dataset.

## 🧹 Outlier Candidate Review

    column = "obligations"

    q1 = df_accounts[column].quantile(0.25)
    q3 = df_accounts[column].quantile(0.75)
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr

    df_outliers = df_accounts[
        (df_accounts[column] < lower_fence) |
        (df_accounts[column] > upper_fence)
    ]

    df_outliers

## ✅ Descriptive Statistics Checklist

Before moving to inferential statistics or modeling:

- Numeric columns have been converted correctly.
- Missing values have been counted.
- Central tendency has been reviewed.
- Dispersion has been reviewed.
- Skewness and kurtosis have been reviewed.
- Outlier candidates have been identified.
- Group-level summaries have been compared.
- Unusual budget values have been validated against source context.
