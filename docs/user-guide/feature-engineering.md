# Feature Engineering

## 🧭 Purpose

This page explains how Sake prepares raw data for machine learning through cleaning, transformation, encoding, scaling, and dimensionality reduction.

Feature engineering converts a source dataset into a model-ready feature matrix while preserving analytical meaning.

## 🧱 Feature Engineering Workflow

| Step | Purpose |
|---|---|
| Column selection | Identify predictors, target, identifiers, and metadata. |
| Numeric coercion | Convert financial values to numeric form. |
| Missing-value handling | Impute, flag, or remove incomplete observations. |
| Categorical encoding | Convert categories to numeric features. |
| Scaling | Standardize numeric magnitude when models require it. |
| Dimensionality reduction | Reduce feature width while preserving useful structure. |
| Feature validation | Confirm outputs remain aligned to source records. |

## 🎯 Separate Features and Target

    target_column = "target"

    df_features = df_accounts.drop(columns=[target_column])
    target = df_accounts[target_column]

Preserve identifiers separately when they should not be used as predictors:

    identifier_columns = ["treasury_account_symbol", "account_name"]

    existing_identifiers = [
        column for column in identifier_columns
        if column in df_features.columns
    ]

    df_identifiers = df_features[existing_identifiers].copy()
    df_features = df_features.drop(columns=existing_identifiers)

## 🔢 Numeric Feature Preparation

Select numeric columns:

    numeric_features = df_features.select_dtypes(include="number").columns.tolist()

Review numeric missing values:

    df_features[numeric_features].isna().sum()

## 🧹 Missing-Value Handling

Simple numeric imputation:

    from sklearn.impute import SimpleImputer
    import pandas as pd

    imputer = SimpleImputer(strategy="median")

    df_numeric = pd.DataFrame(
        imputer.fit_transform(df_features[numeric_features]),
        columns=numeric_features,
        index=df_features.index
    )

Common strategies:

| Strategy | Use |
|---|---|
| Mean | Symmetric numeric distributions. |
| Median | Skewed financial data. |
| Most frequent | Categorical fields. |
| Constant | Explicit missing category or zero-fill when justified. |

## 🧾 Categorical Encoding

Select categorical columns:

    categorical_features = df_features.select_dtypes(include=["object", "category"]).columns.tolist()

One-hot encode categorical columns:

    df_categorical = pd.get_dummies(
        df_features[categorical_features],
        drop_first=False,
        dummy_na=True
    )

## ⚖️ Scaling

Some algorithms benefit from scaling, especially distance-based and regularized models.

    from sklearn.preprocessing import StandardScaler
    import pandas as pd

    scaler = StandardScaler()

    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_numeric),
        columns=df_numeric.columns,
        index=df_numeric.index
    )

Use scaling for:

| Model Family | Scaling Importance |
|---|---|
| Logistic Regression | Often beneficial. |
| Support Vector Machine | Usually important. |
| K-Nearest Neighbors | Important. |
| Linear models with regularization | Important. |
| Tree-based models | Usually not required. |

## 🧮 Combine Feature Blocks

    df_model_features = pd.concat(
        [df_scaled, df_categorical],
        axis=1
    )

## 📉 PCA Example

Principal Component Analysis can reduce dimensionality while preserving variance.

    from sklearn.decomposition import PCA
    import pandas as pd

    pca = PCA(n_components=0.95, random_state=42)

    pca_values = pca.fit_transform(df_scaled)

    df_pca = pd.DataFrame(
        pca_values,
        columns=[f"pc_{index + 1}" for index in range(pca_values.shape[1])],
        index=df_scaled.index
    )

## 📉 Truncated SVD Example

Truncated SVD is useful for sparse or high-dimensional encoded data.

    from sklearn.decomposition import TruncatedSVD
    import pandas as pd

    svd = TruncatedSVD(n_components=10, random_state=42)

    svd_values = svd.fit_transform(df_model_features)

    df_svd = pd.DataFrame(
        svd_values,
        columns=[f"svd_{index + 1}" for index in range(svd_values.shape[1])],
        index=df_model_features.index
    )

## 🏛️ Budget Execution Feature Ideas

| Feature | Purpose |
|---|---|
| Obligation rate | Obligations divided by budgetary resources. |
| Outlay rate | Outlays divided by obligations or resources. |
| Unobligated balance ratio | Balance divided by resources. |
| Recovery ratio | Recoveries divided by obligations. |
| Account category flags | Encoded account type or availability. |
| Fiscal period indicators | Time or reporting-period context. |
| Log-transformed values | Reduce skew in high-dollar financial fields. |

## ✅ Feature Engineering Checklist

Before modeling:

- Target column is separated.
- Identifiers are preserved but excluded from predictors unless justified.
- Numeric fields are converted.
- Missing values are handled.
- Categorical fields are encoded.
- Scaling is applied where appropriate.
- Feature matrix rows align to the target.
- Derived budget ratios are reviewed for divide-by-zero issues.
- Feature names remain interpretable.
