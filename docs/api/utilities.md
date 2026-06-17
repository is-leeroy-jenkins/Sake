# Utilities

## 🧭 Purpose

The `utilities` module should contain shared helpers that support Sake workflows without belonging exclusively to data loading, statistics, features, models, evaluation, or visualization.

Utility functions should remain small, deterministic, and reusable across the application.

## 🧱 Utility Categories

| Category       | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| Validation     | Confirm required values, options, columns, and paths.        |
| Formatting     | Prepare labels, captions, tables, and display values.        |
| Type handling  | Normalize Python, NumPy, Pandas, and model-specific objects. |
| Configuration  | Resolve constants, defaults, and option lists.               |
| Error handling | Standardize user-facing or logged exception information.     |
| Export support | Prepare result tables or artifacts for download.             |

## ✅ Design Expectations

- Keep utilities small and deterministic.
- Avoid hiding workflow-specific business logic in generic helpers.
- Document assumptions in Google-style docstrings.
- Avoid hidden state unless the helper explicitly manages shared configuration.
- Keep function names specific enough to communicate their purpose.

## 📚 Source Reference

::: utilities
