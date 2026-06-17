# Models

## 🧭 Purpose

The `models` module should define or coordinate Sake model construction, training, prediction, comparison, and model-selection workflows for classification and regression tasks.

Model routines should provide consistent inputs and outputs so evaluation, visualization, and benchmarking can operate predictably across multiple estimator families.

## 🧱 Workflow Role

| Area | Responsibility |
|---|---|
| Construction | Create supported estimators and configure model defaults. |
| Training | Fit candidate models to prepared features and targets. |
| Prediction | Generate validation, holdout, or user-selected predictions. |
| Comparison | Return consistent outputs for benchmarking and diagnostics. |
| Configuration | Centralize model options, labels, and reusable settings. |

## ✅ Classification Families

| Family                 | Typical Use                      |
|------------------------|----------------------------------|
| Logistic Regression    | Baseline linear classification.  |
| Support Vector Machine | Margin-based classification.     |
| Decision Tree          | Rule-based classification.       |
| Random Forest          | Ensemble classification.         |
| XGBoost Classifier     | Gradient-boosted classification. |
| K-Nearest Neighbors    | Similarity-based classification. |
| Naive Bayes            | Probabilistic classification.    |

## 📉 Regression Families

| Family                        | Typical Use                          |
|-------------------------------|--------------------------------------|
| Linear Regression             | Baseline numeric prediction.         |
| Ridge, Lasso, ElasticNet      | Regularized linear regression.       |
| Support Vector Regressor      | Margin-based numeric prediction.     |
| Decision Tree Regressor       | Rule-based numeric prediction.       |
| Random Forest Regressor       | Ensemble numeric prediction.         |
| Gradient Boosting / XGBoost   | Boosted numeric prediction.          |
| K-Nearest Neighbors Regressor | Similarity-based numeric prediction. |

## ✅ Model Output Expectations

| Output            | Purpose                                       |
|-------------------|-----------------------------------------------|
| Fitted estimator  | Supports prediction and inspection.           |
| Predictions       | Feed evaluation metrics and diagnostic plots. |
| Training metadata | Supports reproducibility and review.          |
| Timing data       | Supports model benchmarking.                  |
| Model label       | Keeps comparison tables readable.             |

## 📚 Source Reference

::: models
