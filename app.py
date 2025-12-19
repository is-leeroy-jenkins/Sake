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
from typing import List, Dict, Any

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

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ================================================================================================
# Page Configuration
# ================================================================================================
st.set_page_config(page_title="Sake", layout="wide")
st.title("Status of Balances")
st.caption(
    "Machine-Learning Workbench for File A (Account Balances): "
)


# ================================================================================================
# Helper Functions
# ================================================================================================
def styled_table(df: pd.DataFrame, height: int = 420) -> None:
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
        "Normalizer (L2)": Normalizer(norm="l2"),
        "Quantile (Normal)": QuantileTransformer(output_distribution="normal"),
        "Power (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    }[name]


def plot_hist(series: pd.Series, title: str):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(series, bins=50, edgecolor="black", linewidth=0.6)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    st.pyplot(fig)


# ================================================================================================
# Sidebar — Dataset
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
    st.subheader("Dataset Preview")
    styled_table(df.head(50))

    st.subheader("Column Quality")
    quality = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "nulls": df.isna().sum(),
            "unique": df.nunique(),
        }
    )
    styled_table(quality)


# ================================================================================================
# Descriptive Statistics
# ================================================================================================
with tab_desc:
    st.subheader("Descriptive Statistics (Expanded)")

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
    styled_table(desc_df.round(4), height=520)

    st.subheader("Distributions")
    selected = st.multiselect(
        "Select numeric columns",
        numeric_cols,
        default=numeric_cols[: min(6, len(numeric_cols))],
    )

    for col in selected:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        plot_hist(s, f"Distribution of {col}")
        st.caption(
            "Histogram shows spread and shape. Skewed or heavy-tailed distributions "
            "suggest non-normality and potential outliers."
        )


# ================================================================================================
# Inferential Statistics
# ================================================================================================
with tab_inf:
    st.subheader("Inferential Statistics")

    corr = df[numeric_cols].corr()
    styled_table(corr.round(4), height=420)

    st.caption(
        "Correlation values near ±1 indicate strong linear relationships. "
        "Highly correlated features may indicate redundancy."
    )

    pairs = []
    for i, a in enumerate(numeric_cols):
        for b in numeric_cols[i + 1 :]:
            x = pd.to_numeric(df[a], errors="coerce").dropna()
            y = pd.to_numeric(df[b], errors="coerce").dropna()
            n = min(len(x), len(y))
            if n >= 20:
                r, p = stats.pearsonr(x[:n], y[:n])
                pairs.append({"Feature A": a, "Feature B": b, "r": r, "p_value": p, "n": n})

    sig_df = pd.DataFrame(pairs).sort_values("p_value")
    styled_table(sig_df.head(50), height=520)

    st.caption(
        "Low p-values suggest correlations unlikely due to chance. "
        "Effect size (|r|) should be considered alongside significance."
    )


# ================================================================================================
# Feature Analysis (6 Methods)
# ================================================================================================
with tab_feat:
    st.subheader("Feature Analysis")

    methods = st.multiselect(
        "Select Feature Analysis Methods",
        [
            "Scaling Impact",
            "PCA",
            "Truncated SVD",
            "Factor Analysis",
            "LDA (Supervised)",
            "k-Means Clustering",
        ],
        default=["Scaling Impact", "PCA", "k-Means Clustering"],
    )

    X = df[numeric_cols].fillna(0.0)
    scaler = make_scaler(scaler_name)
    Xs = X if scaler is None else pd.DataFrame(scaler.fit_transform(X), columns=numeric_cols)

    if "Scaling Impact" in methods:
        st.markdown("### Scaling Impact")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before Scaling**")
            styled_table(X.describe().T.round(4))
        with c2:
            st.markdown("**After Scaling**")
            styled_table(Xs.describe().T.round(4))

    if "PCA" in methods:
        st.markdown("### PCA")
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=30, edgecolors="black", linewidths=0.4)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        styled_table(
            pd.DataFrame(
                {
                    "Component": ["PC1", "PC2"],
                    "Explained Variance": pca.explained_variance_ratio_,
                }
            ).round(4)
        )

    if "Truncated SVD" in methods:
        st.markdown("### Truncated SVD")
        svd = TruncatedSVD(n_components=2)
        Z = svd.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=30, edgecolors="black", linewidths=0.4)
        ax.grid(alpha=0.25)
        st.pyplot(fig)

    if "Factor Analysis" in methods:
        st.markdown("### Factor Analysis")
        fa = FactorAnalysis(n_components=2)
        Z = fa.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=30, edgecolors="black", linewidths=0.4)
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        styled_table(pd.DataFrame(fa.components_.T, index=numeric_cols, columns=["Factor1", "Factor2"]).round(4))

    if "LDA (Supervised)" in methods:
        st.markdown("### LDA (Supervised)")
        y = (pd.to_numeric(df[target_col], errors="coerce") > df[target_col].median()).astype(int)
        lda = LinearDiscriminantAnalysis(n_components=1)
        Z = lda.fit_transform(Xs, y)
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.scatter(Z[:, 0], np.zeros_like(Z[:, 0]), c=y, cmap="coolwarm", edgecolors="black")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

    if "k-Means Clustering" in methods:
        st.markdown("### k-Means Clustering")
        k = st.slider("Clusters (k)", 2, 8, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)
        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(k):
            idx = labels == i
            ax.scatter(Z[idx, 0], Z[idx, 1], label=f"Cluster {i}", edgecolors="black", linewidths=0.4)
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)
        styled_table(X.assign(cluster=labels).groupby("cluster").mean().round(4))


# ================================================================================================
# Models
# ================================================================================================
with tab_models:
    st.subheader("Model Evaluation")

    task = st.selectbox("Task", ["Regression", "Classification"])

    X_model = df[feature_cols]
    y_raw = df[target_col]

    if task == "Classification":
        y = (pd.to_numeric(y_raw, errors="coerce") > y_raw.median()).astype(int)
        model = RandomForestClassifier(n_estimators=300, random_state=42)
        cv = StratifiedKFold(5)
        scoring = ["accuracy", "precision", "recall", "f1"]
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)
        model = RandomForestRegressor(n_estimators=300, random_state=42)
        cv = KFold(5)
        scoring = ["r2", "neg_mean_absolute_error", "neg_mean_squared_error"]

    preprocessor = ColumnTransformer(
        [
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", make_scaler(scaler_name))]),
             [c for c in feature_cols if c in numeric_cols]),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
             [c for c in feature_cols if c in categorical_cols]),
        ]
    )

    pipe = Pipeline([("prep", preprocessor), ("model", model)])

    if st.button("Run Cross-Validation"):
        scores = cross_validate(pipe, X_model, y, cv=cv, scoring=scoring)
        styled_table(pd.DataFrame(scores).describe().T.round(4))
