# Data Sources

Sake supports budget execution analytics using Account Balances data and related federal financial reporting structures. The documentation in this section defines the source-data concepts, validation expectations, and modeling considerations used throughout the analytical workflow.

## 🧭 Purpose

Document the data sources, financial-reporting context, structural fields, quality controls, and preparation requirements used by Sake.

## 🏛️ Primary Analytical Source

| Source                  | Role                                                                                                                            |
|-------------------------|---------------------------------------------------------------------------------------------------------------------------------|
| File A Account Balances | Provides budgetary account-balance information used for execution review, statistical analysis, and machine-learning workflows. |
| CSV extracts            | Supports local testing, custom analytical subsets, and simplified data movement.                                                |
| Excel workbooks         | Supports manually prepared or exported budget execution datasets.                                                               |
| Pandas DataFrames       | Supports notebook, Colab, Databricks, and programmatic workflows.                                                               |

## 🧾 Account Balances Context

Account Balances data supports review of budgetary resources, obligations, outlays, and available balances across account structures. Sake treats the data as a structured analytical input rather than as a final financial statement.

## 🏦 Treasury Account Symbol Context

Treasury Account Symbols provide the account identity used to associate budget execution activity with federal account structures.

| Field Type              | Analytical Use                                                                    |
|-------------------------|-----------------------------------------------------------------------------------|
| Agency identifier       | Groups execution activity by responsible agency.                                  |
| Main account            | Supports account-level aggregation and comparison.                                |
| Availability period     | Distinguishes annual, multi-year, and no-year resources where represented.        |
| Treasury account symbol | Provides a stable account reference for joins, review, and reporting comparisons. |
| Account title           | Improves human-readable interpretation of account-level outputs.                  |

## 🔄 DATA Act and GTAS Relationship

| Source or Structure | Relationship to Sake                                                                            |
|---------------------|-------------------------------------------------------------------------------------------------|
| DATA Act File A     | Provides budgetary account balance data suitable for analytical review.                         |
| GTAS                | Supports agency trial balance reporting and budgetary-account relationships.                    |
| SF-133              | Provides budget execution reporting context for comparison, reconciliation, and interpretation. |
| TAS crosswalks      | Support mapping between account identifiers and reporting structures.                           |

## 📑 Common Data Categories

| Category                   | Examples                                                                             | Use                                        |
|----------------------------|--------------------------------------------------------------------------------------|--------------------------------------------|
| Account identifiers        | Agency, bureau, main account, TAS, account name.                                     | Grouping, joins, filtering, and reporting. |
| Budget authority           | Budgetary resources and authority-related balances.                                  | Execution profile and resource review.     |
| Obligations                | Obligated amounts and related balances.                                              | Execution pace and commitment analysis.    |
| Outlays                    | Disbursement-related values.                                                         | Spending pattern analysis.                 |
| Unobligated balances       | Available or remaining balances.                                                     | Carryover and execution-risk analysis.     |
| Program or object metadata | Classification, object class, program activity, or related descriptors when present. | Segmentation and feature development.      |

## 🧪 Supported Input Formats

| Format      | Expected Use                                                 | Preparation Notes                                                |
|-------------|--------------------------------------------------------------|------------------------------------------------------------------|
| `.csv`      | Lightweight local data loading and repeatable testing.       | Confirm delimiter, encoding, header row, and numeric formatting. |
| `.xlsx`     | Excel-based Account Balances extracts or analysis workbooks. | Confirm worksheet name, merged cells, and header alignment.      |
| `.xls`      | Legacy Excel input.                                          | Confirm dependency support and column preservation.              |
| `DataFrame` | Notebook, Colab, Databricks, or direct Python workflow.      | Confirm column names, data types, and target-field availability. |

## ✅ Data Quality Controls

| Control                | Purpose                                                                     |
|------------------------|-----------------------------------------------------------------------------|
| Schema validation      | Confirms expected columns exist before analysis or modeling.                |
| Type validation        | Confirms numeric fields can be converted safely.                            |
| Missing-value review   | Identifies incomplete records and fields requiring imputation or exclusion. |
| Duplicate detection    | Prevents double-counting or biased model training.                          |
| Range review           | Flags impossible or unusual financial values.                               |
| Sign convention review | Confirms negative and positive values have expected analytical meaning.     |
| Account-code review    | Confirms account identifiers are populated and usable.                      |
| Outlier review         | Identifies values requiring financial or statistical investigation.         |
| Cardinality review     | Identifies high-cardinality categorical fields before encoding.             |

## 🔢 Numeric Preparation

Budget execution datasets often contain numeric fields stored as strings because of commas, symbols, blanks, annotations, or export formatting.

| Issue            | Handling                                                             |
|------------------|----------------------------------------------------------------------|
| Commas           | Remove formatting before numeric conversion.                         |
| Blank cells      | Convert to missing values before imputation or exclusion.            |
| Parentheses      | Convert accounting-style negative values when applicable.            |
| Percent signs    | Convert only when the field represents a percentage.                 |
| Currency symbols | Remove display symbols while preserving numeric magnitude.           |
| Mixed types      | Coerce with documented error handling and review failed conversions. |

## 🧱 Modeling Considerations

| Consideration                   | Effect                                                                     |
|---------------------------------|----------------------------------------------------------------------------|
| Skewed financial values         | May require transformation, scaling, or robust model selection.            |
| Sparse categories               | May affect encoding and model stability.                                   |
| High-cardinality account fields | May require grouping, hashing, target-safe encoding, or exclusion.         |
| Outliers                        | May reflect valid financial events or data-quality issues.                 |
| Time-period fields              | May require explicit treatment for trend, period, or fiscal-year analysis. |
| Target construction             | Must avoid target leakage and preserve interpretation.                     |

## 🔍 Recommended Data Review Sequence

1. Load the source file or DataFrame.
2. Confirm row count, column count, and column names.
3. Identify account identifiers and reporting fields.
4. Review missing values and malformed records.
5. Convert numeric budget fields.
6. Validate sign conventions and extreme values.
7. Review categorical cardinality.
8. Select features and target fields.
9. Preserve a clean analytical dataset for statistics and modeling.

## 🧾 Data Source Documentation Checklist

- Source file name or extract name is recorded.
- Extract date or reporting period is recorded when available.
- Account identifier fields are preserved.
- Numeric conversion rules are documented.
- Missing-value handling is documented.
- Exclusions or filters are documented.
- Feature and target fields are documented.
- Data-quality warnings are retained for review.
