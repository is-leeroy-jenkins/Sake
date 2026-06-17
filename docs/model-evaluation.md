# Model Evaluation

Sake evaluates classification and regression workflows through consistent metrics, diagnostic outputs, runtime benchmarking, and model-comparison tables. Evaluation results support model selection, analytical interpretation, and repeatable experimentation.

## 🧭 Purpose

Define the metrics, diagnostics, timing outputs, review sequence, and model-comparison standards used by Sake.

## 🔄 Evaluation Workflow

```text
Prepared Features
  -> Train/Test Split or Cross-Validation
  -> Model Training
  -> Prediction
  -> Metric Calculation
  -> Diagnostic Review
  -> Benchmark Comparison
  -> Model Selection
```

## 🧱 Evaluation Inputs

| Input           | Purpose                                                        |
|-----------------|----------------------------------------------------------------|
| Feature matrix  | Provides model-ready predictors.                               |
| Target vector   | Provides classification labels or regression values.           |
| Trained model   | Generates predictions for evaluation.                          |
| Predictions     | Supports metric calculation and diagnostic review.             |
| Validation data | Supports holdout scoring or cross-validation.                  |
| Timing data     | Supports runtime benchmarking and model-comparison review.     |
| Model metadata  | Preserves estimator name, configuration, and workflow context. |

## ✅ Classification Metrics

| Metric                 | Use                                                                  | Interpretation                                                    |
|------------------------|----------------------------------------------------------------------|-------------------------------------------------------------------|
| Accuracy               | Overall correct predictions.                                         | Useful when classes are balanced.                                 |
| Precision              | Correct positive predictions out of predicted positives.             | Useful when false positives are costly.                           |
| Recall                 | Correct positive predictions out of actual positives.                | Useful when false negatives are costly.                           |
| F1 score               | Harmonic mean of precision and recall.                               | Useful when precision and recall must be balanced.                |
| Confusion matrix       | Count of predicted versus actual classes.                            | Shows false positives, false negatives, and class-level behavior. |
| ROC curve              | Threshold performance across true-positive and false-positive rates. | Useful for binary classifier discrimination review.               |
| Precision-recall curve | Positive-class performance across thresholds.                        | Useful for imbalanced classification problems.                    |
| ROC AUC                | Area under the ROC curve.                                            | Summarizes ranking performance.                                   |

## 📉 Regression Metrics

| Metric                  | Use                                       | Interpretation                                                                |
|-------------------------|-------------------------------------------|-------------------------------------------------------------------------------|
| R²                      | Share of variance explained by the model. | Higher values indicate stronger fit, subject to context and overfitting risk. |
| Adjusted R²             | R² adjusted for predictor count.          | Useful when comparing models with different feature counts.                   |
| Mean absolute error     | Average absolute prediction error.        | Easy to interpret in target units.                                            |
| Mean squared error      | Average squared prediction error.         | Penalizes larger errors more heavily.                                         |
| Root mean squared error | Square root of mean squared error.        | Expressed in target units and sensitive to large errors.                      |
| Median absolute error   | Median absolute prediction error.         | Robust to extreme errors.                                                     |
| Residuals               | Actual values minus predicted values.     | Supports review of bias, nonlinearity, and heteroscedasticity.                |

## ⏱️ Runtime Benchmarking

| Timing Measure            | Purpose                                          |
|---------------------------|--------------------------------------------------|
| Fit time                  | Measures training cost.                          |
| Prediction time           | Measures inference cost.                         |
| Total runtime             | Measures end-to-end model execution time.        |
| Cross-validation duration | Measures repeated training and scoring cost.     |
| Visualization time        | Measures chart-generation overhead when tracked. |

## 🧪 Diagnostic Outputs

| Diagnostic                   | Workflow                     | Purpose                                              |
|------------------------------|------------------------------|------------------------------------------------------|
| Confusion matrix             | Classification               | Shows class-level prediction outcomes.               |
| Classification report        | Classification               | Summarizes precision, recall, F1, and support.       |
| ROC curve                    | Classification               | Reviews threshold behavior.                          |
| Precision-recall curve       | Classification               | Reviews positive-class behavior under imbalance.     |
| Predicted-versus-actual plot | Regression                   | Compares predictions to observed values.             |
| Residual plot                | Regression                   | Identifies bias, spread, nonlinearity, and outliers. |
| Error distribution           | Regression                   | Shows prediction error shape.                        |
| Feature importance           | Classification or regression | Identifies influential predictors where supported.   |
| Learning curve               | Classification or regression | Reviews training size effects where implemented.     |

## 🧱 Model Comparison Table

| Column                              | Description                                       |
|-------------------------------------|---------------------------------------------------|
| Model                               | Human-readable estimator name.                    |
| Task                                | Classification or regression.                     |
| Fit time                            | Training duration.                                |
| Prediction time                     | Prediction duration.                              |
| Primary metric                      | Main comparison metric for the task.              |
| Secondary metrics                   | Supporting performance indicators.                |
| Cross-validation mean               | Average score across folds when available.        |
| Cross-validation standard deviation | Stability measure across folds.                   |
| Notes                               | Constraints, warnings, or interpretation details. |

## 🔍 Evaluation Controls

| Control            | Standard                                                                             |
|--------------------|--------------------------------------------------------------------------------------|
| Consistent split   | Compare models using the same train/test split or fold definitions.                  |
| Metric alignment   | Use classification metrics for classification and regression metrics for regression. |
| Target integrity   | Confirm predictions align with true target records.                                  |
| Leakage prevention | Ensure preprocessing does not fit on validation or test data.                        |
| Reproducibility    | Use fixed random seeds where appropriate.                                            |
| Class imbalance    | Review class distribution before relying on accuracy.                                |
| Outlier influence  | Review residuals and large errors before selecting a regression model.               |
| Runtime cost       | Include timing when model speed is relevant.                                         |

## ✅ Recommended Review Sequence

1. Confirm the target field is valid for the selected task.
2. Confirm feature preparation does not leak target information.
3. Train baseline models.
4. Calculate task-appropriate metrics.
5. Review diagnostic plots.
6. Compare fit time, prediction time, and total runtime.
7. Review stability across folds when cross-validation is used.
8. Select a candidate model based on performance, stability, interpretability, and runtime.

## 🧾 Classification Review Checklist

- Class labels are valid.
- Class distribution is reviewed.
- Accuracy is not used alone for imbalanced classes.
- Confusion matrix is reviewed.
- Precision and recall are interpreted against analytical costs.
- ROC or precision-recall behavior is reviewed where available.
- Threshold-dependent behavior is documented when thresholds are adjusted.

## 🧾 Regression Review Checklist

- Target values are numeric and meaningful.
- Predicted-versus-actual plot is reviewed.
- Residual plot is reviewed.
- Error magnitude is interpreted in target units.
- Large residuals are investigated.
- Feature importance or coefficient interpretation is reviewed where supported.
- Overfitting risk is reviewed through validation or cross-validation.

## 🚫 Evaluation Anti-Patterns

| Anti-Pattern                                      | Risk                                                          |
|---------------------------------------------------|---------------------------------------------------------------|
| Comparing models with different train/test splits | Produces unreliable rankings.                                 |
| Using accuracy alone for imbalanced data          | Hides poor minority-class performance.                        |
| Fitting scalers before splitting data             | Creates data leakage.                                         |
| Ignoring residual plots                           | Misses systematic regression errors.                          |
| Selecting only by one metric                      | Ignores operational, runtime, and interpretability tradeoffs. |
| Publishing metrics without data-quality notes     | Separates model results from source-data limitations.         |

## ✅ Output Standard

Evaluation outputs should preserve the model name, task type, metric values, timing values, validation method, and diagnostic artifacts needed to reproduce and interpret the result.
