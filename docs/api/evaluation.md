# Evaluation

## 🧭 Purpose

The `evaluation` module should contain metrics, diagnostics, timing capture, and benchmarking utilities used after Sake trains classification or regression models.

Evaluation functions should make model comparison consistent, reproducible, and interpretable across estimator families.

## 🧱 Workflow Role

| Area        | Responsibility                                                                |
|-------------|-------------------------------------------------------------------------------|
| Metrics     | Calculate task-appropriate classification and regression metrics.             |
| Diagnostics | Package residuals, predicted values, confusion matrices, and scoring outputs. |
| Timing      | Track fit time, prediction time, and total runtime.                           |
| Ranking     | Compare model results across consistent scoring criteria.                     |
| Reporting   | Return model summaries for tables, charts, and downstream review.             |

## 📊 Evaluation Coverage

| Workflow | Metrics and Outputs |
|---|---|
| Classification | Accuracy, precision, recall, F1 score, confusion matrix, ROC review, precision-recall review. |
| Regression | R², mean absolute error, mean squared error, root mean squared error, residual review. |
| Benchmarking | Fit duration, prediction duration, total runtime, score ranking, and fold stability. |

## ✅ Recommended Sequence

1. Confirm predictions align to expected records.
2. Calculate task-specific metrics.
3. Review fit and prediction timing.
4. Compare candidate models using consistent metrics.
5. Inspect diagnostic plots before selecting a preferred model.

## 📚 Source Reference

::: evaluation
