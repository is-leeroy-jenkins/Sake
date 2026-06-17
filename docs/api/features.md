# Features

## 🧭 Purpose

The `features` module should prepare model-ready predictors by selecting columns, encoding categorical fields, scaling numeric values, handling missing values, and applying dimensionality-reduction techniques.

Feature preparation should make model inputs reproducible, explainable, and safe from target leakage.

## 🧱 Workflow Role

| Area              | Responsibility                                                                          |
|-------------------|-----------------------------------------------------------------------------------------|
| Selection         | Identify valid predictor columns and remove unsuitable fields.                          |
| Encoding          | Convert categorical fields into machine-readable features.                              |
| Scaling           | Normalize or standardize numeric fields when required by the model.                     |
| Reduction         | Apply PCA, Truncated SVD, Factor Analysis, or related transformations.                  |
| Target handling   | Prepare supervised-learning targets without leaking target information into predictors. |
| Metadata tracking | Preserve enough feature context for interpretation and diagnostics.                     |

## 🔄 Pipeline Position

```text
Raw DataFrame -> Clean DataFrame -> Encoded Features -> Transformed Matrix -> Model Training
```

## ✅ Safeguards

- Keep target columns out of feature matrices.
- Fit encoders and scalers on training data only.
- Preserve feature names where possible.
- Document dimensionality-reduction configuration.
- Keep transformation metadata available for interpretation.
- Avoid overwriting raw source data during transformation.

## 📚 Source Reference

::: features
