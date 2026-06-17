# Sake Architecture

Sake organizes budget execution analytics into a repeatable workflow for data loading, statistical review, feature preparation, model training, evaluation, and visualization. The architecture separates user-facing interaction from analytical processing so that notebooks, Streamlit pages, and reusable Python modules can share the same core workflow.

## 🧭 Purpose

Define the application structure, workflow boundaries, module responsibilities, and documentation relationship for the Sake machine-learning and statistical-analysis framework.

## 🧱 Architectural Layers

| Layer | Responsibility | Typical Outputs |
|---|---|---|
| User Interface | Coordinates interactive workflows through Streamlit, notebooks, Colab, or Databricks. | Uploaded files, selected workflow options, visible tables, charts, and metrics. |
| Data Layer | Loads Account Balances, CSV, Excel, and DataFrame inputs. | Validated DataFrames, schema summaries, column profiles, and type mappings. |
| Statistics Layer | Performs descriptive and inferential analysis. | Summary tables, correlations, hypothesis-test results, confidence intervals, and outlier indicators. |
| Feature Layer | Converts raw fields into model-ready predictors. | Encoded matrices, scaled features, reduced-dimension components, and selected feature sets. |
| Modeling Layer | Trains and compares classification or regression models. | Fitted estimators, predictions, cross-validation results, and candidate model summaries. |
| Evaluation Layer | Calculates metrics and model diagnostics. | Accuracy, precision, recall, F1, R², error measures, timing, residuals, and rankings. |
| Visualization Layer | Renders charts for interpretation and validation. | Distribution plots, heatmaps, confusion matrices, ROC curves, residual plots, and feature-importance charts. |
| Documentation Layer | Publishes MkDocs pages and source-generated API references. | User guide, API pages, development workflow, and dark-mode site assets. |

## 🔄 Primary Workflow

```text
Source Data
  -> Data Loading
  -> Data Validation
  -> Data Overview
  -> Descriptive Statistics
  -> Inferential Statistics
  -> Feature Engineering
  -> Classification or Regression
  -> Model Evaluation
  -> Visualization
  -> Documentation and Review
```

## 🏛️ Data Flow

| Stage | Input | Processing | Output |
|---|---|---|---|
| Ingestion | File A Account Balances, CSV, Excel, or DataFrame | Read data, normalize column access, preserve source metadata. | Raw analytical DataFrame. |
| Validation | Raw analytical DataFrame | Inspect schema, missing values, numeric fields, duplicates, and categorical cardinality. | Validated DataFrame and quality notes. |
| Profiling | Validated DataFrame | Calculate row counts, column summaries, distributions, and summary statistics. | Data overview and descriptive statistics. |
| Inference | Validated DataFrame and selected variables | Run correlations, group comparisons, and statistical tests. | Inferential results and interpretation support. |
| Transformation | Validated DataFrame and feature selections | Encode, scale, reduce dimensions, and isolate target variables. | Model-ready feature and target arrays. |
| Training | Features, target, and model options | Fit classification or regression estimators. | Fitted models and predictions. |
| Evaluation | Predictions and true values | Calculate metrics, diagnostics, timing, and rankings. | Evaluation tables and diagnostic artifacts. |
| Visualization | Data, metrics, predictions, and diagnostics | Render charts and plots. | Interpretable visual outputs. |

## 🧩 Module Responsibility Model

| Module | Responsibility | Design Boundary |
|---|---|---|
| `app` | User interaction, upload controls, workflow routing, and rendered outputs. | Should not contain reusable statistical or modeling logic that belongs in a lower layer. |
| `data` | Loading, validation, coercion, schema review, and DataFrame preparation. | Should not train models or render final UI components. |
| `statistics` | Descriptive statistics, inferential statistics, correlations, and analytical summaries. | Should not mutate the original source dataset unless explicitly documented. |
| `features` | Feature preparation, encoding, scaling, dimensionality reduction, and target separation. | Should prevent target leakage and preserve transformation metadata. |
| `models` | Estimator creation, training, prediction, and model family coordination. | Should return consistent structures for evaluation and comparison. |
| `evaluation` | Metrics, diagnostics, timing, benchmarking, and model ranking. | Should not retrain models. |
| `visualization` | Plot construction and display-ready figure generation. | Should separate figure construction from business decisions. |
| `utilities` | Shared validation, formatting, configuration, and helper behavior. | Should remain small, deterministic, and reusable. |

## 🧪 Runtime Contexts

| Runtime | Role | Notes |
|---|---|---|
| Streamlit | Interactive application workflow. | Best suited for upload, exploration, model selection, and visual review. |
| Jupyter / Colab | Research and experimentation workflow. | Best suited for iterative analysis, prototype notebooks, and exploratory review. |
| Databricks | Scalable analytical workflow. | Best suited for larger datasets and platform-based experimentation. |
| Local Python | Development and validation workflow. | Best suited for source edits, build checks, and documentation generation. |

## 🔐 Validation Boundaries

| Boundary | Required Control |
|---|---|
| File input | Confirm file type, readable structure, and supported worksheet or table layout. |
| Schema | Verify required columns before statistics, features, or model training. |
| Numeric conversion | Preserve original values where coercion creates missing or invalid numbers. |
| Feature preparation | Prevent target leakage and preserve train/test separation. |
| Model training | Confirm target availability, class balance, and compatible estimator inputs. |
| Evaluation | Use task-appropriate metrics and consistent comparison rules. |
| Visualization | Label charts clearly and avoid implying causality from exploratory plots. |

## 🧾 Documentation Architecture

Sake documentation uses MkDocs, Material for MkDocs, mkdocstrings, Markdown topic pages, dark-mode CSS, and optional JavaScript enhancements.

| Component | Purpose |
|---|---|
| `mkdocs.yml` | Defines site metadata, theme configuration, plugins, extensions, assets, and navigation. |
| `docs/index.md` | Provides the main project landing page. |
| `docs/architecture.md` | Defines the system architecture and workflow responsibilities. |
| `docs/data-sources.md` | Documents Account Balances and related budget execution data concepts. |
| `docs/model-evaluation.md` | Defines evaluation metrics, diagnostics, and benchmarking standards. |
| `docs/user-guide/` | Provides task-oriented operating guidance. |
| `docs/api/` | Provides source-driven API pages through mkdocstrings directives. |
| `docs/development.md` | Defines setup, validation, build, and publishing workflow. |
| `docs/assets/css/sake.css` | Applies dark-mode documentation styling. |
| `docs/assets/js/sake.js` | Adds progressive documentation usability enhancements. |

## ✅ Architecture Review Checklist

- Data loading is separated from UI rendering.
- Statistical routines do not depend on Streamlit state.
- Feature preparation prevents target leakage.
- Model training and evaluation use consistent inputs.
- Evaluation metrics match classification or regression task type.
- Visualization functions receive explicit data or result objects.
- API documentation pages match importable Python modules.
- MkDocs navigation includes every generated documentation page.
