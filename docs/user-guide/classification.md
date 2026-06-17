# Classification

## 🧭 Purpose

This page explains how to train and evaluate classification models in Sake.

Classification models predict categorical outcomes, such as risk category, execution class, high/medium/low status, or another discrete label.

## ✅ Supported Model Families

| Model | Use |
|---|---|
| Logistic Regression | Strong baseline for linear classification. |
| Support Vector Machine | Margin-based classification. |
| Decision Tree | Interpretable rules and splits. |
| Random Forest | Ensemble classification with feature importance. |
| XGBoost Classifier | Gradient-boosted classification. |
| K-Nearest Neighbors | Distance-based classification. |
| Gaussian Naive Bayes | Probabilistic baseline. |
| Extra Trees | Randomized ensemble classification. |
| Bagging | Bootstrap aggregation of base estimators. |
| AdaBoost | Boosted weak learners. |

## 🎯 Target Definition

A classification target should contain discrete categories.

Example:

    target = df_accounts["risk_category"]

The target may be an existing source column or a derived label:

    df_accounts["execution_category"] = pd.cut(
        df_accounts["obligation_rate"],
        bins=[-float("inf"), 0.50, 0.90, float("inf")],
        labels=["Low", "Moderate", "High"]
    )

## 🧪 Train/Test Split

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        df_model_features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target
    )

Use stratification when class distribution should be preserved across train and test sets.

## 🧪 Baseline Model

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

## 📊 Classification Metrics

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(y_test, predictions, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0)
    }

    metrics

## 🔥 Confusion Matrix

    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_test, predictions)
    matrix

A confusion matrix shows where the model correctly and incorrectly assigned categories.

## 🔁 Cross-Validation

    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(
        model,
        df_model_features,
        target,
        cv=5,
        scoring="f1_weighted"
    )

    scores.mean(), scores.std()

## 🏛️ Budget Execution Classification Examples

| Target | Meaning |
|---|---|
| Risk category | Low, medium, or high execution risk. |
| Execution category | Low, moderate, or high obligation rate. |
| Account class | Account type or budget enforcement category. |
| Outlier flag | Normal or anomalous account behavior. |
| Lapse category | Low or high unobligated balance risk. |

## ⚠️ Class Imbalance

Budget data may be imbalanced. For example, very few accounts may represent high-risk observations.

Review class counts:

    target.value_counts(normalize=True)

Possible responses:

| Issue | Response |
|---|---|
| Rare class | Use class weights or resampling. |
| Misleading accuracy | Use precision, recall, and F1. |
| Many categories | Collapse sparse categories when analytically justified. |
| Small samples | Use cross-validation carefully. |

## ✅ Classification Checklist

Before accepting a model:

- Target classes are meaningful.
- Class balance is reviewed.
- Train/test split is reproducible.
- Metrics include more than accuracy.
- Confusion matrix is reviewed.
- Cross-validation results are compared.
- Feature leakage is checked.
- Model outputs are interpretable enough for the intended use.
