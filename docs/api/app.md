# Application

## 🧭 Purpose

The `app` module is the interactive entry point for Sake. It should coordinate the Streamlit interface, data upload flow, workflow selection, statistical analysis controls, feature-engineering options, model-training actions, evaluation output, and visualization rendering.

This page documents the application-facing layer and should remain focused on orchestration rather than duplicating business logic that belongs in data, statistics, features, models, evaluation, or visualization modules.

## 🧱 Workflow Role

| Area             | Responsibility                                                                                         |
|------------------|--------------------------------------------------------------------------------------------------------|
| Page setup       | Configure page layout, title, sidebar controls, and user-facing workflow sections.                     |
| Upload handling  | Accept Account Balances, CSV, Excel, or notebook-provided datasets.                                    |
| Workflow routing | Route user choices to statistics, features, models, evaluation, and visualization modules.             |
| State management | Preserve uploaded data, selected options, transformed features, trained models, and generated outputs. |
| Output rendering | Present previews, metrics, diagnostic plots, warnings, and model-comparison results.                   |

## 🔄 Application Flow

```text
User Input -> Streamlit Controls -> Workflow Routing -> Analytical Module Calls -> Rendered Outputs
```

## ✅ Implementation Expectations

- Keep reusable analytical logic outside the UI layer.
- Avoid unsafe module-level execution that prevents `mkdocstrings` from importing `app`.
- Initialize session-state keys before they are read.
- Use stable labels for Account Balances, budget execution, model type, target field, and evaluation output.
- Keep plotting and metric calculations in dedicated modules when possible.

## 🧪 Validation Checklist

| Check                    | Purpose                                                       |
|--------------------------|---------------------------------------------------------------|
| `python -m compileall .` | Confirm the application module and dependencies compile.      |
| `python -c "import app"` | Confirm `mkdocstrings` can import the module.                 |
| `streamlit run app.py`   | Confirm the interactive interface launches.                   |
| `mkdocs build`           | Confirm the API page renders without import or griffe errors. |

## 📚 Source Reference

::: app
