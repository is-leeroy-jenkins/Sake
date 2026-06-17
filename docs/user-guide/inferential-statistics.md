# Inferential Statistics

## 🧭 Purpose

This page explains how Sake can use inferential statistics to evaluate relationships, group differences, and statistical significance in budget execution or tabular analytical datasets.

Inferential statistics move beyond summary description. They test whether patterns in the data are likely to reflect meaningful relationships or differences rather than random variation.

## 🔍 Common Tests

| Method | Purpose | Example Use |
|---|---|---|
| Pearson correlation | Measures linear association between numeric variables. | Relationship between resources and obligations. |
| Spearman correlation | Measures rank-based monotonic association. | Relationship between account size rank and outlay rank. |
| t-test | Compares means between two groups. | Compare execution rates for two account categories. |
| ANOVA | Compares means across three or more groups. | Compare obligations across availability types. |
| Chi-square test | Tests association between categorical variables. | Determine whether account type relates to risk category. |
| Confidence interval | Estimates plausible range for a statistic. | Estimate expected mean outlays. |
| Regression p-values | Test variable significance in a model. | Determine whether recoveries predict unobligated balance. |

## 📈 Correlation Analysis

Pearson correlation is useful for linear relationships:

    df_corr = df_accounts.select_dtypes(include="number").corr(method="pearson")
    df_corr

Spearman correlation is useful when the relationship is monotonic but not strictly linear:

    df_rank_corr = df_accounts.select_dtypes(include="number").corr(method="spearman")
    df_rank_corr

## 🧪 Two-Sample t-test

Use a t-test when comparing the mean of a numeric field between two groups.

    from scipy import stats

    group_a = df_accounts.loc[df_accounts["account_type"] == "A", "obligations"].dropna()
    group_b = df_accounts.loc[df_accounts["account_type"] == "B", "obligations"].dropna()

    statistic, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)

    statistic, p_value

Use Welch’s t-test by setting `equal_var=False` when equal variance is not assumed.

## 🧪 ANOVA

Use ANOVA when comparing means across more than two groups.

    from scipy import stats

    groups = [
        group["obligations"].dropna()
        for _, group in df_accounts.groupby("availability_type")
    ]

    statistic, p_value = stats.f_oneway(*groups)

    statistic, p_value

## 🧪 Chi-square Test

Use a chi-square test for categorical association.

    from scipy import stats
    import pandas as pd

    df_table = pd.crosstab(
        df_accounts["account_type"],
        df_accounts["risk_category"]
    )

    statistic, p_value, degrees_freedom, expected = stats.chi2_contingency(df_table)

    statistic, p_value, degrees_freedom

## 📐 Confidence Intervals

Estimate the confidence interval for a mean:

    from scipy import stats
    import numpy as np

    values = df_accounts["obligations"].dropna()
    confidence = 0.95

    mean_value = values.mean()
    standard_error = stats.sem(values)
    interval = stats.t.interval(
        confidence,
        len(values) - 1,
        loc=mean_value,
        scale=standard_error
    )

    mean_value, interval

## 🏛️ Budget Execution Interpretation

| Result | Interpretation |
|---|---|
| Low p-value | Evidence suggests the observed difference or relationship is unlikely under the null hypothesis. |
| High p-value | Evidence is insufficient to reject the null hypothesis. |
| Strong correlation | Variables move together, but causation is not established. |
| Wide confidence interval | Estimate is uncertain or data is highly variable. |
| Significant ANOVA | At least one group mean differs; follow-up tests may be needed. |

## ⚠️ Practical Cautions

Inferential tests depend on assumptions. Always consider:

- sample size
- missing values
- independence of observations
- skewed distributions
- outliers
- repeated accounts across time
- account hierarchy
- policy or accounting changes across fiscal periods

## ✅ Inferential Statistics Checklist

Before relying on results:

- Variables are correctly typed.
- Missing values are handled.
- Test assumptions are reviewed.
- Outliers are considered.
- Group definitions are meaningful.
- p-values are interpreted with practical context.
- Correlation is not treated as causation.
- Results are documented with assumptions and limitations.
