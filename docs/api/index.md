# API Reference

## 🧭 Purpose

The Sake API reference documents the source modules that support budget execution analytics, statistical analysis, feature engineering, machine-learning model training, model evaluation, visualization, and shared utilities.

These pages are designed for `mkdocstrings`. Each module page gives a short operational context and then delegates the detailed reference material to the Python source through a source-driven API directive.

## 🧱 API Organization

| Page | Module | Purpose |
|---|---|---|
| [Application](app.md) | `app` | Streamlit application entry point and workflow orchestration. |
| [Data](data.md) | `data` | Data ingestion, validation, coercion, and preparation workflows. |
| [Statistics](statistics.md) | `statistics` | Descriptive statistics, inferential tests, and analytical summaries. |
| [Features](features.md) | `features` | Feature selection, encoding, transformation, and dimensionality reduction. |
| [Models](models.md) | `models` | Classification, regression, training, prediction, and comparison workflows. |
| [Evaluation](evaluation.md) | `evaluation` | Metrics, diagnostics, benchmarking, timing, and post-training review. |
| [Visualization](visualization.md) | `visualization` | Diagnostic charts, model plots, residual plots, confusion matrices, and feature-importance views. |
| [Utilities](utilities.md) | `utilities` | Shared helpers used across the application. |

## 🔄 Documentation Build Position

The API reference sits between the user guide and the source code.

```text
User Guide -> API Reference -> Python Source Docstrings -> Rendered MkDocs Site
```

## ✅ Source Documentation Standard

Use Google-style docstrings for public modules, classes, functions, methods, and properties.

| Section | Use |
|---|---|
| `Purpose:` | Explain the workflow role, state used or modified, and analytical purpose. |
| `Args:` | Document only real parameters from the function signature. |
| `Attributes:` | Document public runtime attributes exposed by classes. |
| `Returns:` | Document meaningful return values with explicit types. |
| `Raises:` | Document validation, data, model, or runtime failures. |
| `Notes:` | Capture constraints, assumptions, and non-obvious behavior. |
| `Examples:` | Provide compact examples when helpful. |

## 🧪 Validation Commands

Run these commands from the repository root before publishing:

```powershell
python -m compileall .
mkdocs build
mkdocs serve
```

## 🛠️ Troubleshooting

| Symptom | Likely Cause | Correction |
|---|---|---|
| `ModuleNotFoundError` | API page references a missing module. | Rename the directive or add the module to the Python path. |
| Griffe parsing warning | Docstring section is malformed. | Repair `Args:`, `Attributes:`, `Returns:`, or `Raises:` formatting. |
| Page missing from nav | File exists but is not listed in `mkdocs.yml`. | Add the page to the `nav` section. |
| Nav references missing file | `mkdocs.yml` points to a file that does not exist. | Create the file or remove the nav entry. |

!!! warning
    These pages assume the matching Python modules exist at the repository root or on the configured `mkdocstrings` Python path. If the project uses different module names, update the matching API directive and `mkdocs.yml` navigation entry.
