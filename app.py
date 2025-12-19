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
from sklearn.neighbors import NearestNeighbors

# ================================================================================================
# Page Config
# ================================================================================================
st.set_page_config(page_title="Sake s", layout="wide")
st.title("Status of Balances")
st.caption("ML Workbench for File A (Account Balances).")

# ================================================================================================
# Helpers
# ================================================================================================
def styled_table(df: pd.DataFrame, height: int = 400) -> None:
    st.dataframe(df, use_container_width=True, height=height)


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
        "Normalizer (L2)": Normalizer(norm="l2"),
        "Quantile (Normal)": QuantileTransformer(output_distribution="normal"),
        "Power (Yeo-Johnson)": PowerTransformer(method="yeo-johnson"),
    }[name]


# ================================================================================================
# Sidebar — Data
# ================================================================================================
st.sidebar.header("Dataset")

upload = st.sidebar.file_uploader("Upload File A (Excel)", type=["xlsx", "xls"])
if not upload:
    st.info("Upload a File A (Account Balances) Excel extract.")
    st.stop()

xls = pd.ExcelFile(io.BytesIO(upload.getvalue()))
sheet = st.sidebar.selectbox("Sheet", xls.sheet_names)
df_raw = pd.read_excel(xls, sheet_name=sheet)
df = df_raw.dropna(how="all").drop_duplicates()

numeric_cols = infer_numeric_columns(df)
target_col = st.sidebar.selectbox("Target Column", numeric_cols)

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
tab_data, tab_feat = st.tabs(["Data", "Feature Analysis"])

# ================================================================================================
# Data Tab
# ================================================================================================
with tab_data:
    styled_table(df.head(50))
    styled_table(
        pd.DataFrame(
            {
                "dtype": df.dtypes.astype(str),
                "non-null": df.notna().sum(),
                "nulls": df.isna().sum(),
                "unique": df.nunique(),
            }
        )
    )

# ================================================================================================
# Feature Analysis
# ================================================================================================
with tab_feat:
    st.subheader("Feature Analysis")

    selected_methods = st.multiselect(
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

    # --------------------------------------------------------------------------------------------
    if "Scaling Impact" in selected_methods:
        st.markdown("### Feature Scaling Impact")

        before = X.describe().T[["mean", "std", "min", "max"]]
        after = Xs.describe().T[["mean", "std", "min", "max"]]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Before Scaling**")
            styled_table(before.round(4))
        with c2:
            st.markdown("**After Scaling**")
            styled_table(after.round(4))

        st.caption(
            "Scaling alters feature magnitude and variance without changing rank order. "
            "Models relying on distance or gradient descent are highly sensitive to this step."
        )

    # --------------------------------------------------------------------------------------------
    if "PCA" in selected_methods:
        st.markdown("### Principal Component Analysis (PCA)")
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=28, edgecolors="black", linewidths=0.4)
        ax.set_title("PCA — First Two Components")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        evr = pd.DataFrame(
            {
                "component": ["PC1", "PC2"],
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )
        styled_table(evr.round(4))

        st.caption(
            "PCA identifies orthogonal directions capturing maximum variance. "
            "Strong concentration in early components indicates correlated features."
        )

    # --------------------------------------------------------------------------------------------
    if "Truncated SVD" in selected_methods:
        st.markdown("### Truncated SVD")

        svd = TruncatedSVD(n_components=2)
        Z = svd.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=28, edgecolors="black", linewidths=0.4)
        ax.set_title("Truncated SVD Projection")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        styled_table(
            pd.DataFrame(
                {"component": ["SVD1", "SVD2"], "explained_variance_ratio": svd.explained_variance_ratio_}
            ).round(4)
        )

        st.caption(
            "SVD provides PCA-like decomposition without requiring centering, "
            "making it suitable for sparse or large-scale matrices."
        )

    # --------------------------------------------------------------------------------------------
    if "Factor Analysis" in selected_methods:
        st.markdown("### Factor Analysis")

        fa = FactorAnalysis(n_components=2)
        Z = fa.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(Z[:, 0], Z[:, 1], s=28, edgecolors="black", linewidths=0.4)
        ax.set_title("Latent Factor Space")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        loadings = pd.DataFrame(fa.components_.T, index=numeric_cols, columns=["Factor1", "Factor2"])
        styled_table(loadings.round(4))

        st.caption(
            "Factor Analysis models shared latent drivers rather than maximizing variance. "
            "Useful when features are believed to arise from hidden budgetary mechanisms."
        )

    # --------------------------------------------------------------------------------------------
    if "LDA (Supervised)" in selected_methods:
        st.markdown("### Linear Discriminant Analysis (Supervised)")

        y = (pd.to_numeric(df[target_col], errors="coerce") > df[target_col].median()).astype(int)

        lda = LinearDiscriminantAnalysis(n_components=1)
        Z = lda.fit_transform(Xs, y)

        fig, ax = plt.subplots(figsize=(7, 3))
        ax.scatter(Z[:, 0], np.zeros_like(Z[:, 0]), c=y, cmap="coolwarm", edgecolors="black")
        ax.set_title("LDA — Class Separation")
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        st.caption(
            "LDA maximizes class separability. Clear separation indicates strong predictive structure "
            "between features and the target variable."
        )

    # --------------------------------------------------------------------------------------------
    if "k-Means Clustering" in selected_methods:
        st.markdown("### k-Means Clustering")

        k = st.slider("Number of clusters", 2, 8, 3)
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)

        pca = PCA(n_components=2)
        Z = pca.fit_transform(Xs)

        fig, ax = plt.subplots(figsize=(7, 4))
        for i in range(k):
            idx = labels == i
            ax.scatter(
                Z[idx, 0],
                Z[idx, 1],
                label=f"Cluster {i}",
                s=28,
                edgecolors="black",
                linewidths=0.4,
            )
        ax.set_title("k-Means Clusters (PCA Space)")
        ax.legend()
        ax.grid(alpha=0.25)
        st.pyplot(fig)

        cluster_means = X.assign(cluster=labels).groupby("cluster").mean()
        styled_table(cluster_means.round(4))

        st.caption(
            "Clusters group accounts with similar numeric profiles. "
            "Cluster means provide interpretable summaries of each group’s behavior."
        )
