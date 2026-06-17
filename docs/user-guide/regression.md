# Regression

## 🧭 Purpose

This page explains how to train and evaluate regression models in Sake.

Regression models predict numeric outcomes, such as obligations, outlays, balances, execution rates, recoveries, or other continuous financial measures.

## ✅ Supported Model Families

| Model | Use |
|---|---|
| Linear Regression | Traditional baseline numeric prediction. |
| Ridge Regression | L2-regularized linear regression. |
| Lasso Regression | L1-regularized feature selection and regression. |
| ElasticNet | Combined L1 and L2 regularization. |
| Support Vector Regressor | Margin-based numeric prediction. |
| Decision Tree Regressor | Interpretable non-linear regression. |
| Random Forest Regressor | Ensemble non-linear regression. |
| Gradient Boosting Regressor | Sequential boosted regression. |
| XGBoost Regressor | Gradient-boosted regression. |
| K-Nearest Neighbors Regressor | Distance-based numeric prediction. |
| AdaBoost Regressor | Boosted weak learner regression. |
| Extra Trees Regressor | Randomized ensemble regression. |

## 🎯 Target Definition

A regression target should be numeric.

    target = df_accounts["outlays"]

Common budget execution targets include:

| Target | Meaning |
|---|---|
| Obligations | Amount legally committed. |
| Outlays | Amount disbursed. |
| Unobligated balance | Available amount not obligated. |
| Execution rate | Obligations divided by resources. |
| Outlay rate | Outlays divided by obligations or resources. |

## 🧪 Train/Test Split

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df_model_features,
        target,
        test_size=0.20,
        random_state=42
    )

## 🧪 Baseline Model

    from sklearn.linear_model import LinearRegression

    model = LinearRegression()

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

## 📊 Regression Metrics

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    import numpy as np

    metrics = {
        "r2": r2_score(y_test, predictions),
        "mae": mean_absolute_error(y_test, predictions),
        "mse": mean_squared_error(y_test, predictions),
        "rmse": np.sqrt(mean_squared_error(y_test, predictions))
    }

    metrics

## 📈 Predicted Versus Actual

A predicted-versus-actual chart helps evaluate fit and bias.

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted Versus Actual")
    plt.show()

## 🎭 Residual Analysis

Residuals show prediction error.

    residuals = y_test - predictions

    plt.figure()
    plt.scatter(predictions, residuals)
    plt.axhline(0)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals Versus Predicted")
    plt.show()

Look for:

| Pattern | Interpretation |
|---|---|
| Random spread around zero | Model errors are reasonably stable. |
| Funnel shape | Heteroskedasticity or scale effect. |
| Curved pattern | Missing non-linear structure. |
| Extreme residuals | Outliers or influential observations. |
| Systematic positive/negative residuals | Bias in the model. |

## 🔁 Cross-Validation

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        model,
        df_model_features,
        target,
        cv=5,
        scoring="r2"
    )

    scores.mean(), scores.std()

## 🏛️ Budget Execution Regression Notes

Budget data is often skewed. Consider:

- log-transforming large dollar values
- winsorizing only when analytically justified
- separating account types before modeling
- preserving fiscal-year or period context
- checking whether policy changes affect comparability
- validating predictions against source reporting logic

## ✅ Regression Checklist

Before accepting a model:

- Target variable is numeric.
- Extreme values are reviewed.
- Train/test split is reproducible.
- R², MAE, MSE, and RMSE are reviewed.
- Residual plots are inspected.
- Cross-validation is performed.
- Feature leakage is checked.
- Results are interpreted in budget execution context.
