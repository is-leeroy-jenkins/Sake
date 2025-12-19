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
import time
from typing import Any, Dict, List, Optional

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
)
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)

from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ================================================================================================
# Page Config
# ================================================================================================
st.set_page_config(
    page_title="Sake",
    layout="wide",
)

st.title("Status of Balances")
st.caption(
    "Interactive analysis and modeling workbench for File A (Account Balances): "
    "descriptive statistics, inferential testing, feature analysis, and ML evaluation."
)


# ================================================================================================
# Helpers
# ================================================================================================
def styled_table(df: pd.DataFrame, height: int = 400) -> None:
    st.dataframe(df, use_container_width=True, height=height)


def infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
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
        "Normalizer L2": Normalizer(norm="l2"),
        "Quantile (Normal)": QuantileTransformer(output_distribution="normal"),
        "Power (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    }[name]


def plot_hist(values, title, xlabel):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(values, bins=50, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    st.pyplot(fig)


# ================================================================================================
# Sidebar — Data
# ================================================================================================
st.sidebar.header("Dataset")

upload = st.sidebar.file_uploader("Upload File A (Excel)", type=["xlsx", "xls"])
if not upload:
    st.info("Upload a File A (Account Balances) Excel extract to begin.")
    st.stop()

xls = pd.ExcelFile(io.BytesIO(upload.getvalue()))
sheet = st.sidebar.selectbox("Sheet", xls.sheet_names)
df_raw = pd.read_excel(xls, sheet_name=sheet)

df = df_raw.dropna(how="all").drop_duplicates()

numeric_cols = infer_numeric_columns(df)
categorical_cols = [c for c in df.columns if c not in numeric_cols]

target_col = st.sidebar.selectbox("Target Column", numeric_cols)
feature_cols = st.sidebar.multiselect(
    "Feature Columns",
    options=[c for c in df.columns if c != target_col],
    default=[c for c in numeric_cols if c != target_col],
)

scaler_name = st.sidebar.selectbox(
    "Feature Scaling",
    ["None", "Standard", "Robust", "MinMax", "MaxAbs", "Normalizer L2", "Quantile (Normal)", "Power (Yeo-Johnson)"],
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
    st.subheader("Preview")
    styled_table(df.head(50))

    st.subheader("Column Quality")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non-null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
        }
    )
    styled_table(summary)


# ================================================================================================
# Descriptive Statistics (Expanded)
# ================================================================================================
with tab_desc:
    st.subheader("Descriptive Statistics")

    rows: List[Dict[str, Any]] = []
    for col in numeric_cols:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        rows.append(
            {
                "feature": col,
                "mean": s.mean(),
                "std": s.std(),
                "min": s.min(),
                "q25": s.quantile(0.25),
                "median": s.median(),
                "q75": s.quantile(0.75),
                "max": s.max(),
                "skew": stats.skew(s),
                "kurtosis": stats.kurtosis(s),
            }
        )

    desc_df = pd.DataFrame(rows).set_index("feature")
    styled_table(desc_df.round(4), height=500)

    st.subheader("Distributions")
    selected = st.multiselect("Select numeric columns", numeric_cols, default=numeric_cols[:5])

    for col in selected:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        plot_hist(s, f"Distribution of {col}", col)
        st.caption(
            "This histogram shows the distribution and spread of values. "
            "Skewed or heavy-tailed shapes suggest non-normality and potential outliers."
        )


# ================================================================================================
# Inferential Statistics (Expanded)
# ================================================================================================
with tab_inf:
    st.subheader("Inferential Statistics")

    st.markdown("### Correlation Significance")
    corr = df[numeric_cols].corr()
    styled_table(corr.round(4))

    st.caption(
        "Correlation values near ±1 indicate strong linear relationships. "
        "High correlations suggest redundancy or shared drivers."
    )

    st.markdown("### Pairwise Correlation Tests")
    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            x = pd.to_numeric(df[a], errors="coerce").dropna()
            y = pd.to_numeric(df[b], errors="coerce").dropna()
            n = min(len(x), len(y))
            if n >= 20:
                r, p = stats.pearsonr(x[:n], y[:n])
                pairs.append({"A": a, "B": b, "r": r, "p-value": p, "n": n})

    sig_df = pd.DataFrame(pairs).sort_values("p-value")
    styled_table(sig_df.head(40), height=500)

    st.caption(
        "Low p-values indicate correlations unlikely due to random chance. "
        "Effect size (|r|) should be considered alongside statistical significance."
    )


# ================================================================================================
# Feature Analysis
# ================================================================================================
with tab_feat:
    st.subheader("Feature Analysis")

    X = df[numeric_cols].fillna(0.0).values
    scaler = make_scaler(scaler_name)
    Xs = X if scaler is None else scaler.fit_transform(X)

    st.markdown("### PCA")
    pca = PCA(n_components=2)
    Z = pca.fit_transform(Xs)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(Z[:, 0], Z[:, 1], s=25, edgecolors="black", linewidths=0.4)
    ax.set_title("PCA — First Two Components")
    ax.grid(alpha=0.25)
    st.pyplot(fig)

    st.caption(
        "PCA projects the data into orthogonal components capturing maximum variance. "
        "Tight clusters indicate correlated structure; wide spread suggests multiple drivers."
    )

    st.markdown("### k-Means Clustering")
    k = st.slider("Number of clusters (k)", 2, 8, 3)
    km = KMeans(n_clusters=k, n_init=10)
    labels = km.fit_predict(Xs)

    cluster_means = (
        pd.DataFrame(X, columns=numeric_cols)
        .assign(cluster=labels)
        .groupby("cluster")
        .mean()
    )
    styled_table(cluster_means.round(3))

    st.caption(
        "Cluster means describe typical profiles for each group. "
        "Use these to label clusters operationally (e.g., high outlays vs high unobligated balances)."
    )


# ================================================================================================
# Models
# ================================================================================================
with tab_models:
    st.subheader("Model Evaluation")

    task = st.selectbox("Task", ["Regression", "Classification"])

    X = df[feature_cols]
    y_raw = df[target_col]

    if task == "Classification":
        y = (pd.to_numeric(y_raw, errors="coerce") > y_raw.median()).astype(int)
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        splitter = StratifiedKFold(5)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        splitter = KFold(5)

    scaler = make_scaler(scaler_name)
    num_pipe = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler) if scaler else ("passthrough", "passthrough"),
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline(num_pipe), [c for c in feature_cols if c in numeric_cols]),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             [c for c in feature_cols if c in categorical_cols]),
        ]
    )

    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    run = st.button("Train & Evaluate")

    if run:
        scores = cross_validate(pipe, X, y, cv=splitter, scoring=None, return_train_score=False)
        styled_table(pd.DataFrame(scores).describe().T.round(4))

        st.caption(
            "Cross-validation summarizes model stability across folds. "
            "Large variance suggests sensitivity to data splits or overfitting."
        )
