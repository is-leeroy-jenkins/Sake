# Visualization

## 🧭 Purpose

This page explains the visualization outputs used in Sake to inspect data quality, understand distributions, evaluate models, and communicate results.

Visual review is an important complement to statistical metrics. A model can report acceptable summary metrics while still showing bias, outliers, unstable residuals, or poor performance for important classes.

## 📊 Visualization Categories

| Category | Purpose |
|---|---|
| Data profile charts | Inspect distributions, missing values, and outliers. |
| Statistical charts | Compare groups and relationships. |
| Classification diagnostics | Review categorical prediction performance. |
| Regression diagnostics | Review numeric prediction performance. |
| Feature charts | Identify influential predictors or reduced-dimension structure. |
| Benchmark charts | Compare model performance across algorithms. |

## 📈 Distribution Plots

Histograms show the distribution of numeric fields.

    import matplotlib.pyplot as plt

    column = "obligations"

    plt.figure()
    df_accounts[column].dropna().hist(bins=30)
    plt.xlabel(column)
    plt.ylabel("Count")
    plt.title(f"Distribution of {column}")
    plt.show()

Use distribution plots to identify skew, zeros, negative values, and extreme values.

## 📦 Boxplots

Boxplots help identify outliers and compare groups.

    column = "obligations"

    plt.figure()
    df_accounts.boxplot(column=column)
    plt.title(f"Boxplot of {column}")
    plt.show()

Grouped boxplot:

    df_accounts.boxplot(column="obligations", by="account_type")
    plt.title("Obligations by Account Type")
    plt.suptitle("")
    plt.show()

## 🔥 Correlation Heatmap

    import matplotlib.pyplot as plt

    df_corr = df_accounts.select_dtypes(include="number").corr()

    plt.figure()
    plt.imshow(df_corr)
    plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=90)
    plt.yticks(range(len(df_corr.columns)), df_corr.columns)
    plt.colorbar()
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

## ✅ Classification Visuals

| Visual | Use |
|---|---|
| Confusion matrix | Shows correct and incorrect class predictions. |
| ROC curve | Shows tradeoff between true-positive and false-positive rates. |
| Precision-recall curve | Useful when positive classes are rare. |
| Class distribution chart | Shows class imbalance. |
| Feature importance chart | Shows influential predictors for supported models. |

### Confusion Matrix Example

    from sklearn.metrics import ConfusionMatrixDisplay

    ConfusionMatrixDisplay.from_predictions(y_test, predictions)
    plt.title("Confusion Matrix")
    plt.show()

## 📉 Regression Visuals

| Visual | Use |
|---|---|
| Predicted versus actual | Shows fit and systematic bias. |
| Residuals versus predicted | Shows error structure. |
| Residual histogram | Shows error distribution. |
| Feature importance | Shows influential predictors. |
| Benchmark bar chart | Compares model metrics. |

### Predicted Versus Actual Example

    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted Versus Actual")
    plt.show()

### Residual Histogram Example

    residuals = y_test - predictions

    plt.figure()
    residuals.hist(bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.show()

## 🧮 Model Benchmark Chart

    df_results.plot(
        kind="bar",
        x="model",
        y="score",
        legend=False
    )

    plt.ylabel("Score")
    plt.title("Model Benchmark")
    plt.tight_layout()
    plt.show()

## 🏛️ Budget Execution Visualization Notes

Budget data often includes very large outliers. Consider:

- log-scale views for high-dollar variables
- separate views by account type
- showing top accounts by obligation or outlay
- reviewing negative values separately
- labeling charts clearly with fiscal year or reporting period
- avoiding misleading axes for financial data

## ✅ Visualization Checklist

Before using charts in reports:

- Chart title describes the measure and population.
- Axes are labeled.
- Units are clear.
- Outliers are not hidden without explanation.
- Group definitions are documented.
- Model diagnostics support the metric interpretation.
- The chart is readable in dark-mode documentation.
