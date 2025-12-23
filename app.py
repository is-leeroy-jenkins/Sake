from __future__ import annotations

# ================================================================================================
# Imports (ALL imports live here — no late imports anywhere)
# ================================================================================================
import io
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy import stats

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer,
)
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# ================================================================================================
# Page Config (MUST be first Streamlit call)
# ================================================================================================
st.set_page_config(
    page_title="Sake — Status of Balances",
    page_icon="🍶",
    layout="wide",
)

st.title("Sake — Status of Balances")
st.caption(
    "Audited analytical workbench for File A (Account Balances): "
    "descriptive statistics, inferential testing, feature analysis, and modeling."
)

# ================================================================================================
# Helper Functions
# ================================================================================================
def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
        else:
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() >= 0.95:
                cols.append(c)
    return cols


def make_scaler(name: str):
    return {
        "None": None,
        "Standard": StandardScaler(),
        "Robust": RobustScaler(),
        "MinMax": MinMaxScaler(),
        "MaxAbs": MaxAbsScaler(),
        "Normalizer (L2)": Normalizer(),
        "Quantile (Normal)": QuantileTransformer(output_distribution="normal"),
        "Power (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    }[name]


def safe_kruskal(groups: List[np.ndarray]) -> float | None:
    if len(groups) < 2:
        return None
    if np.allclose(np.concatenate(groups), groups[0][0]):
        return None
    if all(np.var(g) == 0 for g in groups):
        return None
    try:
        return stats.kruskal(*groups).pvalue
    except Exception:
        return None


# ================================================================================================
# Sidebar — Dataset (Upload + Fallback) [FINAL — PATH SAFE]
# ================================================================================================
APP_ROOT = Path(__file__).parent.resolve()
DATA_FALLBACK_PATH = APP_ROOT / "data" / "sample_account_balances.xlsx"
fallback_available = DATA_FALLBACK_PATH.exists()

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload File A (Excel)",
    type=["xlsx", "xls"],
)

# Checkbox is ALWAYS rendered if fallback exists
use_fallback = False
if fallback_available:
    use_fallback = st.sidebar.checkbox(
        "Use bundled sample dataset",
        value=True,
        help=f"Loads sample dataset from {DATA_FALLBACK_PATH}",
    )

# Unified loading logic
if uploaded_file is not None:
    xls = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    source_label = f"Uploaded file: {uploaded_file.name}"

elif fallback_available and use_fallback:
    xls = pd.ExcelFile(DATA_FALLBACK_PATH)
    source_label = f"Fallback file: {DATA_FALLBACK_PATH.name}"

else:
    st.info("Please upload a dataset or enable the bundled sample dataset.")
    st.stop()

st.sidebar.caption(f"📄 Data source: {source_label}")

sheet = st.sidebar.selectbox(
    "Sheet",
    options=xls.sheet_names,
    index=0,
)

df_raw = pd.read_excel(xls, sheet_name=sheet)

# --------------------------------------------------------------------------------
# Unified Excel loading (NO premature st.stop)
# --------------------------------------------------------------------------------
if uploaded_file is not None:
    xls = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    source_label = f"Uploaded file: {uploaded_file.name}"

elif fallback_available and use_fallback:
    xls = pd.ExcelFile(DATA_FALLBACK_PATH)
    source_label = f"Fallback file: {DATA_FALLBACK_PATH.name}"

else:
    st.info("Please upload a dataset or enable the bundled sample dataset.")
    st.stop()

st.sidebar.caption(f"📄 Data source: {source_label}")

sheet = st.sidebar.selectbox(
    "Sheet",
    options=xls.sheet_names,
    index=0,
)

df_raw = pd.read_excel(xls, sheet_name=sheet)


# ================================================================================================
# Canonical Dataset Preparation (UNCONDITIONAL)
# ================================================================================================
df = df_raw.dropna(how="all").drop_duplicates()

numeric_cols = infer_numeric_columns(df)
categorical_cols = [c for c in df.columns if c not in numeric_cols]

# ================================================================================================
# Task & Target (UNCONDITIONAL)
# ================================================================================================
task = st.sidebar.selectbox(
    "Task Type",
    ["Regression", "Classification"],
)

target_options = numeric_cols if task == "Regression" else df.columns.tolist()

target_col = st.sidebar.selectbox(
    "Target Column",
    options=target_options,
)

# ================================================================================================
# Feature Scaling (UNCONDITIONAL)
# ================================================================================================
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
        "Power (Yeo-Johnson)",
    ],
    index=1,
)

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
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Numeric (auto)", f"{len(numeric_cols):,}")

    st.dataframe(df.head(50), use_container_width=True)

# ================================================================================================
# Descriptive Statistics
# ================================================================================================
with tab_desc:
    desc = df[numeric_cols].describe().T
    desc["skew"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    st.dataframe(desc.round(4), use_container_width=True)

# ================================================================================================
# Inferential Statistics (Defensive)
# ================================================================================================
with tab_inf:
    st.subheader("Correlation Matrix")
    corr = df[numeric_cols].corr()
    st.dataframe(corr.round(4), use_container_width=True)

    st.subheader("Target-Aware Tests")

    y_series = df[target_col].dropna()

    if task == "Classification":
        groups = {}
        for label in y_series.unique():
            groups[label] = df.loc[y_series == label, numeric_cols]

        rows = []
        for col in numeric_cols:
            samples = [
                g[col].dropna().values
                for g in groups.values()
                if g[col].dropna().shape[0] >= 5
            ]
            p = safe_kruskal(samples)
            if p is not None:
                rows.append({"feature": col, "kruskal_p": p})

        if rows:
            st.dataframe(pd.DataFrame(rows).sort_values("kruskal_p"), use_container_width=True)
        else:
            st.info("No valid inferential tests could be performed.")

# ================================================================================================
# Feature Analysis
# ================================================================================================
with tab_feat:
    X = df[numeric_cols].fillna(0.0)
    scaler = make_scaler(scaler_name)
    Xs = X if scaler is None else pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)

    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots()
    ax.scatter(Z[:, 0], Z[:, 1], s=25, edgecolor="black")
    ax.set_title("PCA Projection")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

# ================================================================================================
# Models (SAFE)
# ================================================================================================
with tab_models:
    features = st.multiselect(
        "Feature Columns",
        options=[c for c in df.columns if c != target_col],
        default=numeric_cols,
    )

    X_model = df[features]
    y_model = df[target_col]

    if task == "Classification":
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        cv = StratifiedKFold(5)
        scoring = ["accuracy", "precision", "recall", "f1"]
    else:
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        cv = KFold(5)
        scoring = ["r2", "neg_mean_absolute_error"]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", make_scaler(scaler_name)),
            ]), [c for c in features if c in numeric_cols]),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), [c for c in features if c in categorical_cols]),
        ]
    )

    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    if st.button("Run Cross-Validation"):
        scores = cross_validate(pipe, X_model, y_model, cv=cv, scoring=scoring)
        st.dataframe(pd.DataFrame(scores).describe().T.round(4), use_container_width=True)
