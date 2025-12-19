'''
  ******************************************************************************************
      Assembly:                Name
      Filename:                name.py
      Author:                  Terry D. Eppler
      Created:                 05-31-2022

      Last Modified By:        Terry D. Eppler
      Last Modified On:        05-01-2025
  ******************************************************************************************
  <copyright file="guro.py" company="Terry D. Eppler">

	     name.py
	     Copyright ©  2022  Terry Eppler

     Permission is hereby granted, free of charge, to any person obtaining a copy
     of this software and associated documentation files (the “Software”),
     to deal in the Software without restriction,
     including without limitation the rights to use,
     copy, modify, merge, publish, distribute, sublicense,
     and/or sell copies of the Software,
     and to permit persons to whom the Software is furnished to do so,
     subject to the following conditions:

     The above copyright notice and this permission notice shall be included in all
     copies or substantial portions of the Software.

     THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
     INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
     FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
     IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
     ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
     DEALINGS IN THE SOFTWARE.

     You can contact me at:  terryeppler@gmail.com or eppler.terry@epa.gov

  </copyright>
  <summary>
    name.py
  </summary>
  ******************************************************************************************
'''
from __future__ import annotations

import io
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.linear_model import LogisticRegression, Ridge, RidgeClassifier, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ================================================================================================
# Streamlit Page Config
# ================================================================================================
st.set_page_config(page_title="Sake", layout="wide")
st.title("Status of Balances")
st.caption(
    "Analytics and Modeling Workbench for File A (Account Balances): "
)


# ================================================================================================
# Utilities
# ================================================================================================
def styled_table(df: pd.DataFrame, height: int = 420) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
            continue

        coerced = pd.to_numeric(df[c], errors="coerce")
        if coerced.notna().mean() >= 0.95:
            cols.append(c)

    return cols


def infer_categorical_columns(df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
    numeric_set = set(numeric_cols)
    return [c for c in df.columns if c not in numeric_set]


def safe_quantile_transformer(n_samples: int, output_distribution: str) -> QuantileTransformer:
    # QuantileTransformer requires n_quantiles <= n_samples.
    n_quantiles = int(min(1000, max(10, n_samples)))
    return QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles, random_state=0)


def make_scaler(name: str, n_samples: int):
    # All options here are feasible given numeric data, with guard for QuantileTransformer.
    if name == "None":
        return None
    if name == "Standard":
        return StandardScaler()
    if name == "Robust":
        return RobustScaler()
    if name == "MinMax":
        return MinMaxScaler()
    if name == "MaxAbs":
        return MaxAbsScaler()
    if name == "Normalizer (L2)":
        return Normalizer(norm="l2")
    if name == "Quantile (Normal)":
        return safe_quantile_transformer(n_samples=n_samples, output_distribution="normal")
    if name == "Quantile (Uniform)":
        return safe_quantile_transformer(n_samples=n_samples, output_distribution="uniform")
    if name == "Power (Yeo-Johnson)":
        return PowerTransformer(method="yeo-johnson", standardize=True)

    raise ValueError(f"Unknown scaler: {name}")


def plot_hist(values: np.ndarray, title: str, xlabel: str, bins: int = 50) -> None:
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.hist(values, bins=bins, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_scatter(x: np.ndarray, y: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.scatter(x, y, s=30, edgecolors="black", linewidths=0.4, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    st.pyplot(fig, clear_figure=True)


def plot_scatter_by_label(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    uniq = np.unique(labels)
    markers = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

    for i, u in enumerate(uniq):
        idx = labels == u
        ax.scatter(
            x[idx],
            y[idx],
            s=34,
            marker=markers[i % len(markers)],
            edgecolors="black",
            linewidths=0.4,
            alpha=0.85,
            label=str(u),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(title="Group", loc="best")
    st.pyplot(fig, clear_figure=True)


def cramers_v(confusion: np.ndarray) -> float:
    # Cramér’s V for categorical association
    chi2 = stats.chi2_contingency(confusion, correction=False)[0]
    n = confusion.sum()
    if n == 0:
        return float("nan")
    r, k = confusion.shape
    denom = n * (min(k - 1, r - 1))
    if denom <= 0:
        return float("nan")
    return float(math.sqrt(chi2 / denom))


def encode_classification_target(y: pd.Series, numeric_binning: Optional[str]) -> Tuple[np.ndarray, str, Dict[int, str]]:
    """
    Purpose:
        Convert an arbitrary target column to classification labels.

    Parameters:
        y: Target series.
        numeric_binning: If y is numeric-like, optionally bin to:
            - None: treat numeric as categorical via label encoding (many classes possible)
            - "Binary (median)": bin into {0,1} based on median
            - "Quantiles (4)": bin into quartiles (0..3)

    Returns:
        labels: Encoded integer labels.
        mode: A short label describing how it was encoded.
        label_map: Mapping from integer label -> original label (stringified).
    """
    y_num = pd.to_numeric(y, errors="coerce")
    numeric_like = y_num.notna().mean() >= 0.95

    if numeric_like and numeric_binning == "Binary (median)":
        med = float(np.nanmedian(y_num.values))
        labels = (y_num.fillna(med) > med).astype(int).values
        label_map = {0: "≤ median", 1: "> median"}
        return labels, "numeric→binary(median)", label_map

    if numeric_like and numeric_binning == "Quantiles (4)":
        q = pd.qcut(y_num.fillna(y_num.median()), q=4, labels=False, duplicates="drop")
        labels = q.astype(int).values
        label_map = {int(i): f"Q{i+1}" for i in sorted(np.unique(labels))}
        return labels, "numeric→quartiles(4)", label_map

    # Default: treat as categorical labels (including codes)
    y_str = y.astype(str).fillna("NA")
    le = LabelEncoder()
    labels = le.fit_transform(y_str.values)
    label_map = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
    return labels, "categorical(label-encoded)", label_map


@dataclass
class ModelRunResult:
    task: str
    model_name: str
    fit_time_s: float
    metrics_table: pd.DataFrame
    cv_table: Optional[pd.DataFrame]


# ================================================================================================
# Sidebar: Dataset Upload
# ================================================================================================
st.sidebar.header("Dataset")
upload = st.sidebar.file_uploader("Upload File A (Excel)", type=["xlsx", "xls"])

if not upload:
    st.info("Upload a File A (Account Balances) Excel extract to begin.")
    st.stop()

xls = pd.ExcelFile(io.BytesIO(upload.getvalue()))
sheet = st.sidebar.selectbox("Sheet", xls.sheet_names, index=0)
df_raw = pd.read_excel(xls, sheet_name=sheet)

# Conservative cleaning; avoids losing legitimate zeros
df = df_raw.dropna(how="all").drop_duplicates()

numeric_cols = infer_numeric_columns(df)
categorical_cols = infer_categorical_columns(df, numeric_cols)

st.sidebar.subheader("Task & Target (Consistency Fix)")

task = st.sidebar.selectbox("Task", ["Regression", "Classification"], index=1)

if task == "Regression":
    target_options = numeric_cols
else:
    target_options = list(df.columns)

if not target_options:
    st.error("No valid target columns available based on the selected task.")
    st.stop()

target_col = st.sidebar.selectbox("Target Column", target_options, index=0)

numeric_binning = None
if task == "Classification":
    # Only relevant if target is numeric-like
    numeric_binning = st.sidebar.selectbox(
        "If target is numeric-like: binning strategy",
        ["None (label-encode raw values)", "Binary (median)", "Quantiles (4)"],
        index=1,
    )
    if numeric_binning.startswith("None"):
        numeric_binning = None

st.sidebar.subheader("Features")
feature_cols = st.sidebar.multiselect(
    "Feature Columns",
    options=[c for c in df.columns if c != target_col],
    default=[c for c in numeric_cols if c != target_col][: min(12, max(1, len(numeric_cols) - 1))],
)

st.sidebar.subheader("Scaling (numeric features)")
scaler_name = st.sidebar.selectbox(
    "Feature Scaling",
    [
        "None",
        "Standard",
        "Robust",
        "MinMax",
        "MaxAbs",
        "Normalizer (L2)",
        "Quantile (Normal)",
        "Quantile (Uniform)",
        "Power (Yeo-Johnson)",
    ],
    index=1,
)

st.sidebar.subheader("Split / CV")
test_size = st.sidebar.slider("Test size", 0.10, 0.50, 0.25, 0.05)
random_state = st.sidebar.number_input("Random state", 0, 10_000, 42, 1)
enable_cv = st.sidebar.checkbox("Enable cross-validation", value=True)
cv_folds = st.sidebar.slider("CV folds", 3, 10, 5, 1)


# ================================================================================================
# Tabs
# ================================================================================================
tab_data, tab_desc, tab_inf, tab_feat, tab_models = st.tabs(
    ["Data", "Descriptive Statistics", "Inferential Statistics", "Feature Analysis", "Models"]
)


# ================================================================================================
# Data Tab
# ================================================================================================
with tab_data:
    st.subheader("Dataset Preview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric (auto)", f"{len(numeric_cols):,}")
    c4.metric("Categorical (auto)", f"{len(categorical_cols):,}")

    styled_table(df.head(50), height=380)

    st.subheader("Column Quality")
    quality = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "null_%": (df.isna().mean() * 100).round(2),
            "unique": df.nunique(dropna=True),
        }
    ).sort_values("null_%", ascending=False)
    styled_table(quality, height=520)


# ================================================================================================
# Descriptive Statistics
# ================================================================================================
with tab_desc:
    st.subheader("Descriptive Statistics (Expanded)")

    if numeric_cols:
        rows: List[Dict[str, Any]] = []
        for col in numeric_cols:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "feature": col,
                    "count": int(s.size),
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if s.size > 1 else float("nan"),
                    "min": float(s.min()),
                    "q10": float(s.quantile(0.10)),
                    "q25": float(s.quantile(0.25)),
                    "median": float(s.median()),
                    "q75": float(s.quantile(0.75)),
                    "q90": float(s.quantile(0.90)),
                    "max": float(s.max()),
                    "skew": float(stats.skew(s.values, bias=False)) if s.size > 2 else float("nan"),
                    "kurtosis": float(stats.kurtosis(s.values, bias=False)) if s.size > 3 else float("nan"),
                }
            )

        desc_df = pd.DataFrame(rows).set_index("feature").sort_index()
        styled_table(desc_df.round(4), height=560)

        st.subheader("Distributions (multi-select)")
        selected = st.multiselect(
            "Select numeric columns to plot",
            options=numeric_cols,
            default=numeric_cols[: min(6, len(numeric_cols))],
        )

        bins = st.slider("Histogram bins", 10, 200, 50, 5)

        for col in selected:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if s.empty:
                continue
            plot_hist(s.values, f"Histogram — {col}", col, bins=bins)
            st.caption(
                "Interpretation: skewed or heavy-tailed shapes suggest non-normality and potential outliers. "
                "If models are sensitive to scale, consider Robust/Quantile/Power scaling."
            )
    else:
        st.info("No numeric columns were detected in this dataset.")


# ================================================================================================
# Inferential Statistics
# ================================================================================================
with tab_inf:
    st.subheader("Inferential Statistics (Expanded, task-consistent)")

    # Numeric ↔ Numeric correlations are always valid when numeric_cols exist
    if numeric_cols and len(numeric_cols) >= 2:
        st.markdown("### Numeric Correlation Matrix")
        corr_kind = st.selectbox("Correlation method", ["pearson", "spearman"], index=0)
        corr = df[numeric_cols].apply(pd.to_numeric, errors="coerce").corr(method=corr_kind)
        styled_table(corr.round(4), height=420)
        st.caption(
            "High absolute correlation suggests redundancy/shared signal. Consider PCA, regularization, or "
            "dropping one of the variables if multicollinearity is problematic."
        )

        st.markdown("### Correlation Significance (Top Pairs)")
        pairs: List[Dict[str, Any]] = []
        for i, a in enumerate(numeric_cols):
            for b in numeric_cols[i + 1 :]:
                x = pd.to_numeric(df[a], errors="coerce").dropna()
                y = pd.to_numeric(df[b], errors="coerce").dropna()
                n = min(len(x), len(y))
                if n < 20:
                    continue
                x = x.iloc[:n].values
                y = y.iloc[:n].values
                if corr_kind == "pearson":
                    r, p = stats.pearsonr(x, y)
                else:
                    r, p = stats.spearmanr(x, y)
                pairs.append({"Feature A": a, "Feature B": b, "corr": float(r), "p_value": float(p), "n": int(n)})

        if pairs:
            sig_df = pd.DataFrame(pairs).sort_values(["p_value", "corr"], ascending=[True, False]).head(50)
            styled_table(sig_df, height=520)
            st.caption(
                "Low p-values indicate correlations unlikely due to chance. Always consider effect size (|corr|) "
                "and practical relevance. Large datasets can make small effects 'significant'."
            )

    # Target-aware inferential blocks
    st.markdown("### Target-Aware Tests")

    y_series = df[target_col]

    if task == "Regression":
        # Regression target must be numeric; enforce
        y_num = pd.to_numeric(y_series, errors="coerce")
        if y_num.notna().mean() < 0.95:
            st.warning("Selected regression target is not numeric-like; choose a numeric target.")
        else:
            st.caption(
                "Regression target: numeric. Below shows correlation of numeric features with the target "
                "and simple significance tests."
            )
            rows = []
            for c in numeric_cols:
                if c == target_col:
                    continue
                x = pd.to_numeric(df[c], errors="coerce").dropna()
                y = y_num.dropna()
                n = min(len(x), len(y))
                if n < 20:
                    continue
                r, p = stats.pearsonr(x.iloc[:n].values, y.iloc[:n].values)
                rows.append({"feature": c, "pearson_r_to_target": float(r), "p_value": float(p), "n": int(n)})

            if rows:
                tdf = pd.DataFrame(rows).sort_values("p_value").head(50)
                styled_table(tdf, height=520)
                st.caption(
                    "Interpretation: features with strong |r| and low p-values may explain variance in the target. "
                    "Use feature analysis and modeling to validate robustness."
                )
            else:
                st.info("Insufficient non-null overlap to compute feature↔target correlations.")
    else:
        # Classification: allow categorical targets
        labels, mode, label_map = encode_classification_target(y_series, numeric_binning=numeric_binning)

        st.caption(
            f"Classification target encoding: {mode}. "
            "Below are association tests between target and features."
        )

        # Numeric features by categorical target: ANOVA + Kruskal
        if numeric_cols:
            st.markdown("#### Numeric Features by Class (ANOVA & Kruskal-Wallis)")
            use_numeric = st.multiselect(
                "Numeric columns to test",
                options=numeric_cols,
                default=numeric_cols[: min(8, len(numeric_cols))],
                key="inf_num_by_class",
            )
            
            rows = [ ]
            
            for col in use_numeric:
	            x = pd.to_numeric( df[ col ], errors="coerce" )
	            
	            groups = [ ]
	            variances = [ ]
	            
	            for k in np.unique( labels ):
		            vals = x[ labels == k ].dropna( ).values
		            if len( vals ) >= 5:
			            groups.append( vals )
			            variances.append( np.var( vals ) )
	            
	            # Preconditions
	            if len( groups ) < 2:
		            continue
	            
	            if np.allclose( np.concatenate( groups ), groups[ 0 ][ 0 ] ):
		            # All values identical across all groups
		            continue
	            
	            if np.all( np.array( variances ) == 0 ):
		            # No within-group variance anywhere
		            continue
	            
	            try:
		            fstat, p_anova = stats.f_oneway( *groups )
	            except Exception:
		            p_anova = np.nan
	            
	            try:
		            kw = stats.kruskal( *groups )
		            p_kw = kw.pvalue
	            except Exception:
		            p_kw = np.nan
	            
	            rows.append(
		            {
				            "feature": col,
				            "anova_p": p_anova,
				            "kruskal_p": p_kw,
				            "groups_used": len( groups ),
		            }
	            )

            if rows:
                out = pd.DataFrame(rows).sort_values(["kruskal_p", "anova_p"]).head(50)
                styled_table(out, height=520)
                st.caption(
                    "ANOVA compares class means (parametric). Kruskal-Wallis is rank-based and more robust to "
                    "skew/outliers. Prefer Kruskal when distributions are non-normal."
                )
            else:
                st.info("Not enough per-class samples to compute ANOVA/Kruskal for the selected numeric features.")

        # Categorical feature association: Chi-square + Cramér’s V
        if categorical_cols:
            st.markdown("#### Categorical Features vs Class (Chi-square & Cramér’s V)")
            use_cat = st.multiselect(
                "Categorical columns to test",
                options=categorical_cols,
                default=categorical_cols[: min(8, len(categorical_cols))],
                key="inf_cat_by_class",
            )

            rows = []
            for col in use_cat:
                x = df[col].astype(str).fillna("NA")
                ct = pd.crosstab(x, labels)
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p, _, _ = stats.chi2_contingency(ct.values)
                v = cramers_v(ct.values)
                rows.append({"feature": col, "chi2_p": float(p), "cramers_v": float(v), "levels": int(ct.shape[0])})

            if rows:
                out = pd.DataFrame(rows).sort_values(["chi2_p", "cramers_v"], ascending=[True, False]).head(50)
                styled_table(out, height=520)
                st.caption(
                    "Cramér’s V measures strength of association (0..1). Low chi-square p-values suggest the feature "
                    "distribution differs by class."
                )
            else:
                st.info("No valid categorical association tests computed (insufficient levels/classes).")


# ================================================================================================
# Feature Analysis
# ================================================================================================
with tab_feat:
    st.subheader("Feature Analysis (No impossible options)")

    if not numeric_cols:
        st.info("Feature analysis requires numeric features. No numeric columns were detected.")
    else:
        methods = st.multiselect(
            "Select Feature Analysis Methods",
            [
                "Scaling Impact",
                "PCA",
                "Truncated SVD",
                "Factor Analysis",
                "LDA (Supervised) — classification only",
                "k-Means Clustering",
                "Mutual Information — classification only",
            ],
            default=["Scaling Impact", "PCA", "k-Means Clustering"],
        )

        X_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        scaler = make_scaler(scaler_name, n_samples=len(X_num))
        Xs = X_num.values if scaler is None else scaler.fit_transform(X_num.values)

        if "Scaling Impact" in methods:
            st.markdown("### Scaling Impact")
            before = X_num.describe().T
            after = pd.DataFrame(Xs, columns=numeric_cols).describe().T
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before scaling**")
                styled_table(before.round(4), height=380)
            with c2:
                st.markdown("**After scaling**")
                styled_table(after.round(4), height=380)
            st.caption(
                "Scaling changes feature magnitude/variance. Distance-based methods (k-means) and many optimizers "
                "become more stable after scaling."
            )

        if "PCA" in methods:
            st.markdown("### PCA (2D)")
            pca = PCA(n_components=2, random_state=0)
            Z = pca.fit_transform(Xs)
            plot_scatter(Z[:, 0], Z[:, 1], "PCA — Component 1 vs 2", "PC1", "PC2")
            evr = pd.DataFrame(
                {
                    "component": ["PC1", "PC2"],
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                }
            )
            styled_table(evr.round(4), height=220)
            st.caption(
                "PCA captures maximum variance in orthogonal components. Strong concentration in early components "
                "suggests correlated features."
            )

        if "Truncated SVD" in methods:
            st.markdown("### Truncated SVD (2D)")
            svd = TruncatedSVD(n_components=2, random_state=0)
            Z = svd.fit_transform(Xs)
            plot_scatter(Z[:, 0], Z[:, 1], "Truncated SVD — Component 1 vs 2", "SVD1", "SVD2")
            if hasattr(svd, "explained_variance_ratio_"):
                evr = pd.DataFrame(
                    {"component": ["SVD1", "SVD2"], "explained_variance_ratio": svd.explained_variance_ratio_}
                )
                styled_table(evr.round(4), height=220)
            st.caption(
                "SVD provides PCA-like structure and can be preferable in some high-dimensional settings."
            )

        if "Factor Analysis" in methods:
            st.markdown("### Factor Analysis (2D + Loadings)")
            fa = FactorAnalysis(n_components=2, random_state=0)
            Z = fa.fit_transform(Xs)
            plot_scatter(Z[:, 0], Z[:, 1], "Factor Analysis — Factor 1 vs 2", "Factor1", "Factor2")
            loadings = pd.DataFrame(fa.components_.T, index=numeric_cols, columns=["Factor1", "Factor2"])
            styled_table(loadings.round(4), height=520)
            st.caption(
                "Factor analysis models shared latent drivers. Loadings indicate which features contribute most "
                "to each latent factor."
            )

        # LDA is only valid for classification with >=2 classes and numeric features
        if "LDA (Supervised) — classification only" in methods:
            st.markdown("### LDA (Supervised)")
            if task != "Classification":
                st.info("LDA is only available when Task = Classification.")
            else:
                labels, mode, label_map = encode_classification_target(df[target_col], numeric_binning=numeric_binning)
                n_classes = len(np.unique(labels))
                if n_classes < 2:
                    st.info("LDA requires at least 2 classes.")
                else:
                    # n_components must be <= min(n_features, n_classes - 1); we limit to 2 for plotting clarity.
                    max_components = int(min(2, len(numeric_cols), n_classes - 1))
                    if max_components < 1:
                        st.info("LDA cannot be computed with the current target/classes.")
                    else:
                        n_comp = st.selectbox("LDA components", list(range(1, max_components + 1)), index=0)
                        lda = LinearDiscriminantAnalysis(n_components=n_comp)
                        Z = lda.fit_transform(Xs, labels)

                        if Z.shape[1] == 1:
                            jitter = np.random.RandomState(0).normal(0, 0.02, size=len(Z))
                            plot_scatter_by_label(
                                Z[:, 0],
                                jitter,
                                labels,
                                f"LDA — 1D projection (encoding: {mode})",
                                "LDA1",
                                "(jitter)",
                            )
                        else:
                            plot_scatter_by_label(
                                Z[:, 0],
                                Z[:, 1],
                                labels,
                                f"LDA — 2D projection (encoding: {mode})",
                                "LDA1",
                                "LDA2",
                            )

                        st.caption(
                            "LDA maximizes between-class separation. Cleaner separation implies the numeric feature set "
                            "contains strong signal for predicting the chosen target."
                        )

        if "k-Means Clustering" in methods:
            st.markdown("### k-Means Clustering (numeric features)")
            k = st.slider("Clusters (k)", 2, 10, 3, 1)
            km = KMeans(n_clusters=k, n_init=10, random_state=0)
            clusters = km.fit_predict(Xs)

            # Visualize clusters in PCA space for interpretability
            pca2 = PCA(n_components=2, random_state=0)
            Z = pca2.fit_transform(Xs)
            plot_scatter_by_label(Z[:, 0], Z[:, 1], clusters, "k-Means clusters in PCA space", "PC1", "PC2")

            prof = pd.DataFrame(X_num.values, columns=numeric_cols).assign(cluster=clusters).groupby("cluster").mean()
            styled_table(prof.round(4).reset_index(), height=420)
            st.caption(
                "Cluster means provide a profile of typical accounts/rows in each cluster. Use these to label clusters "
                "operationally (e.g., high outlays vs high unobligated balances)."
            )

        # Mutual information is only valid for classification targets (discrete labels)
        if "Mutual Information — classification only" in methods:
            st.markdown("### Mutual Information (numeric features → classification target)")
            if task != "Classification":
                st.info("Mutual information is only available when Task = Classification.")
            else:
                labels, mode, label_map = encode_classification_target(df[target_col], numeric_binning=numeric_binning)
                mi = mutual_info_classif(Xs, labels, random_state=0, discrete_features=False)
                mi_df = pd.DataFrame({"feature": numeric_cols, "mutual_information": mi}).sort_values(
                    "mutual_information", ascending=False
                )
                styled_table(mi_df.round(6), height=520)
                st.caption(
                    "Mutual information measures dependency (including nonlinear). Higher values suggest the feature "
                    "contains more information about the target classes."
                )


# ================================================================================================
# Models
# ================================================================================================
with tab_models:
    st.subheader("Models (task-consistent; no impossible actions)")

    if not feature_cols:
        st.warning("Select at least one feature column in the sidebar.")
        st.stop()

    X = df[feature_cols].copy()

    # Build preprocessing for features:
    num_features = [c for c in feature_cols if c in numeric_cols]
    cat_features = [c for c in feature_cols if c in categorical_cols]

    scaler = make_scaler(scaler_name, n_samples=len(df))

    num_steps: List[Tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scaler is not None:
        num_steps.append(("scaler", scaler))

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=num_steps), num_features),
            ("cat", cat_pipe, cat_features),
        ],
        remainder="drop",
    )

    if task == "Regression":
        y = pd.to_numeric(df[target_col], errors="coerce").fillna(0.0)

        model_name = st.selectbox(
            "Model",
            ["RandomForestRegressor", "LinearRegression", "Ridge"],
            index=0,
        )

        if model_name == "RandomForestRegressor":
            model = RandomForestRegressor(n_estimators=400, random_state=int(random_state), n_jobs=None)
        elif model_name == "Ridge":
            model = Ridge(alpha=1.0, random_state=int(random_state))
        else:
            model = LinearRegression()

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        run = st.button("Train & Evaluate", type="primary")
        if run:
            t0 = time.perf_counter()

            if enable_cv:
                cv = KFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))
                scores = cross_validate(
                    pipe,
                    X,
                    y,
                    cv=cv,
                    scoring=["r2", "neg_mean_absolute_error", "neg_mean_squared_error"],
                    return_train_score=False,
                )
                cv_df = pd.DataFrame(scores)
                cv_df["test_mae"] = -cv_df["test_neg_mean_absolute_error"]
                cv_df["test_mse"] = -cv_df["test_neg_mean_squared_error"]
                cv_df["test_rmse"] = np.sqrt(cv_df["test_mse"])

                styled_table(cv_df.round(6), height=320)
                styled_table(cv_df.describe().T.round(6), height=420)

                st.caption(
                    "Regression CV: higher R² and lower MAE/RMSE indicate better predictive accuracy. "
                    "Large variance across folds suggests instability or heterogeneity."
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=int(random_state)
                )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                metrics = pd.DataFrame(
                    [
                        {"metric": "R²", "value": float(r2_score(y_test, y_pred))},
                        {"metric": "MAE", "value": float(mean_absolute_error(y_test, y_pred))},
                        {"metric": "MSE", "value": float(mean_squared_error(y_test, y_pred))},
                        {"metric": "RMSE", "value": float(np.sqrt(mean_squared_error(y_test, y_pred)))},
                    ]
                )
                styled_table(metrics, height=220)

            t1 = time.perf_counter()
            st.caption(f"Elapsed: {t1 - t0:.2f}s")

    else:
        y_labels, mode, label_map = encode_classification_target(df[target_col], numeric_binning=numeric_binning)
        n_classes = len(np.unique(y_labels))

        st.caption(f"Classification target encoding: {mode} | Classes: {n_classes}")

        # Models that are always feasible:
        model_name = st.selectbox(
            "Model",
            ["RandomForestClassifier", "LogisticRegression", "RidgeClassifier"],
            index=0,
        )

        if model_name == "RandomForestClassifier":
            model = RandomForestClassifier(n_estimators=400, random_state=int(random_state), n_jobs=None)
        elif model_name == "RidgeClassifier":
            model = RidgeClassifier()
        else:
            # LogisticRegression supports multi-class; ensure solver supports it.
            model = LogisticRegression(max_iter=2000, multi_class="auto")

        pipe = Pipeline(steps=[("prep", preprocessor), ("model", model)])

        run = st.button("Train & Evaluate", type="primary")
        if run:
            t0 = time.perf_counter()

            if enable_cv:
                cv = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=int(random_state))
                scores = cross_validate(
                    pipe,
                    X,
                    y_labels,
                    cv=cv,
                    scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
                    return_train_score=False,
                )
                cv_df = pd.DataFrame(scores)
                styled_table(cv_df.round(6), height=320)
                styled_table(cv_df.describe().T.round(6), height=420)

                st.caption(
                    "Classification CV (macro metrics): macro-averaging treats each class equally, which is appropriate "
                    "when class imbalance exists and you want balanced performance."
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_labels,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    stratify=y_labels if n_classes >= 2 else None,
                )
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)

                acc = float(accuracy_score(y_test, y_pred))
                p, r, f, _ = precision_recall_fscore_support(
                    y_test, y_pred, average="macro", zero_division=0
                )

                metrics = pd.DataFrame(
                    [
                        {"metric": "Accuracy", "value": acc},
                        {"metric": "Precision (macro)", "value": float(p)},
                        {"metric": "Recall (macro)", "value": float(r)},
                        {"metric": "F1 (macro)", "value": float(f)},
                    ]
                )
                styled_table(metrics.round(6), height=220)

                st.markdown("#### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(6.2, 5.0))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax, colorbar=False)
                ax.set_title("Confusion Matrix")
                st.pyplot(fig, clear_figure=True)
                st.caption("Diagonal cells are correct classifications; off-diagonals show confusions between classes.")

                st.markdown("#### Classification Report")
                st.code(classification_report(y_test, y_pred, zero_division=0), language="text")

                # ROC / PR are only meaningful and feasible for binary classification with probability outputs
                if n_classes == 2 and hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)[:, 1]
                        st.markdown("#### ROC Curve (binary only)")
                        fig, ax = plt.subplots(figsize=(6.2, 4.6))
                        RocCurveDisplay.from_predictions(y_test, proba, ax=ax)
                        ax.set_title("ROC Curve")
                        st.pyplot(fig, clear_figure=True)

                        st.markdown("#### Precision–Recall Curve (binary only)")
                        fig, ax = plt.subplots(figsize=(6.2, 4.6))
                        PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax)
                        ax.set_title("Precision–Recall Curve")
                        st.pyplot(fig, clear_figure=True)

                        st.caption(
                            "ROC/PR curves are displayed only for binary targets and only when the model exposes "
                            "probabilities. This avoids offering metrics that cannot be computed."
                        )
                    except Exception:
                        st.info("ROC/PR curves unavailable for this model/fit (probabilities not accessible).")

            t1 = time.perf_counter()
            st.caption(f"Elapsed: {t1 - t0:.2f}s")
