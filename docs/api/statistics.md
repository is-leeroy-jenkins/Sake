# Statistics

## 🧭 Purpose

The `statistics` module should provide descriptive and inferential statistical utilities used to understand budget execution data, Account Balances values, modeling inputs, and analytical relationships.

Statistical functions should help users distinguish routine execution patterns from anomalies, group differences, correlations, skewed distributions, and model-relevant relationships.

## 📊 Descriptive Statistics

| Statistic           | Use                                                |
|---------------------|----------------------------------------------------|
| Mean                | Average value across records.                      |
| Median              | Robust central tendency for skewed values.         |
| Mode                | Most common value.                                 |
| Standard deviation  | Spread around the mean.                            |
| Variance            | Squared dispersion.                                |
| Range               | Difference between maximum and minimum.            |
| Interquartile range | Spread of the middle 50 percent of observations.   |
| Skewness            | Direction and magnitude of distribution asymmetry. |
| Kurtosis            | Tail weight and peak behavior.                     |

## 🔍 Inferential Statistics

| Test or Metric       | Use                                            |
|----------------------|------------------------------------------------|
| Pearson correlation  | Linear relationship between numeric variables. |
| Spearman correlation | Rank-based monotonic relationship.             |
| t-test               | Difference between two group means.            |
| ANOVA                | Difference among multiple group means.         |
| Chi-square test      | Relationship among categorical values.         |
| Confidence interval  | Estimated range around a statistic.            |
| Regression p-values  | Significance of explanatory variables.         |
| Z-score              | Distance from the mean in standard deviations. |

## ✅ Recommended Sequence

1. Inspect distributions, missing values, and data types.
2. Calculate descriptive statistics.
3. Review correlations and outliers.
4. Select inferential tests based on variable type.
5. Interpret results against budget execution context.

## 📚 Source Reference

::: statistics
