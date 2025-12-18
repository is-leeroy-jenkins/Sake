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
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt

from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

from sklearn.linear_model import LinearRegression, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.cluster import KMeans, DBSCAN


# -------------------------------------------------------------------------------------------------
# Streamlit App Configuration
# -------------------------------------------------------------------------------------------------
st.set_page_config(
    page_title="Sake — Status of Balances (ML Workbench)",
    layout="wide",
)

st.title("Sake — Status of Balances")
st.caption(
    "A Streamlit workbench derived from the Sake notebook/README: load File A (Account Balances), "
    "run descriptive/inferential analysis, feature engineering, and a unified model evaluation pipeline."
)


# -------------------------------------------------------------------------------------------------
# Defaults aligned with the notebook’s numeric columns
# -------------------------------------------------------------------------------------------------
DEFAULT_NUMERIC_COLUMNS: List[str] = [
    "CarryoverAuthority",
    "CarryoverAdjustments",
    "AnnualAppropriations",
    "BorrowingAuthority",
    "ContractAuthority",
    "OffsettingReceipts",
    "Obligations",
    "Recoveries",
    "UnobligatedBalance",
    "Outlays",
    "TotalResources",
]


# -------------------------------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------------------------------
def _safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _infer_numeric_columns(df: pd.DataFrame) -> List[str]:
    candidates: List[str] = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
            continue

        # Heuristic: attempt to coerce to numeric and see if it mostly works
        coerced = pd.to_numeric(df[col], errors="coerce")
        ok_ratio = float(coerced.notna().mean()) if len(coerced) else 0.0
        if ok_ratio >= 0.95:
            candidates.append(col)

    return candidates


def _infer_categorical_columns(df: pd.DataFrame, numeric_cols: Iterable[str]) -> List[str]:
    numeric_set = set(numeric_cols)
    cats: List[str] = []
    for col in df.columns:
        if col in numeric_set:
            continue
        cats.append(col)
    return cats


def _to_binary_by_median(y: pd.Series) -> pd.Series:
    y_num = pd.to_numeric(y, errors="coerce").fillna(0.0)
    med = float(y_num.median())
    return (y_num > med).astype(int)


def _fmt_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"


@dataclass
class TrainResult:
    model_name: str
    task: str
    fit_time: float
    predict_time: float
    metrics: Dict[str, Any]
    cv_metrics: Optional[pd.DataFrame]


# -------------------------------------------------------------------------------------------------
# Data Loading
# -------------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_excel(upload: bytes, sheet_name: str) -> pd.DataFrame:
    bio = io.BytesIO(upload)
    df = pd.read_excel(bio, sheet_name=sheet_name)
    return df


@st.cache_data(show_spinner=False)
def list_excel_sheets(upload: bytes) -> List[str]:
    bio = io.BytesIO(upload)
    xls = pd.ExcelFile(bio)
    return list(xls.sheet_names)


# -------------------------------------------------------------------------------------------------
# Unified pipeline (train_and_evaluate) per README concept
# -------------------------------------------------------------------------------------------------
def train_and_evaluate(
    model: Any,
    model_name: str,
    task: str,
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    categorical_cols: List[str],
    numeric_cols: List[str],
    test_size: float,
    random_state: int,
    scaler: str,
    do_cv: bool,
    cv_folds: int,
) -> TrainResult:
    X = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Task-specific target shaping
    if task == "classification":
        y = _to_binary_by_median(y_raw)
    elif task == "regression":
        y = pd.to_numeric(y_raw, errors="coerce").fillna(0.0)
    else:
        raise ValueError(f"Unsupported task: {task}")

    # Preprocessing
    if scaler == "standard":
        num_scaler = StandardScaler()
    elif scaler == "robust":
        num_scaler = RobustScaler()
    else:
        num_scaler = None

    numeric_transform = []
    numeric_transform.append(("imputer", SimpleImputer(strategy="median")))
    if num_scaler is not None:
        numeric_transform.append(("scaler", num_scaler))

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=numeric_transform),
                [c for c in numeric_cols if c in feature_cols],
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                [c for c in categorical_cols if c in feature_cols],
            ),
        ],
        remainder="drop",
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if task == "classification" else None
    )

    # Fit timing
    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    t1 = time.perf_counter()

    # Predict timing
    p0 = time.perf_counter()
    y_pred = pipe.predict(X_test)
    p1 = time.perf_counter()

    fit_time = t1 - t0
    predict_time = p1 - p0

    metrics: Dict[str, Any] = {}

    if task == "regression":
        metrics["R²"] = float(r2_score(y_test, y_pred))
        metrics["MAE"] = float(mean_absolute_error(y_test, y_pred))
        metrics["MSE"] = float(mean_squared_error(y_test, y_pred))
        metrics["RMSE"] = float(np.sqrt(metrics["MSE"]))
    else:
        metrics["Accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["Precision"] = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["Recall"] = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["F1"] = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["Report"] = classification_report(y_test, y_pred, zero_division=0, output_dict=False)

    # Optional CV
    cv_df: Optional[pd.DataFrame] = None
    if do_cv:
        if task == "classification":
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = {"accuracy": "accuracy", "f1": "f1"}
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            scoring = {"r2": "r2", "neg_mae": "neg_mean_absolute_error"}

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=None,
            return_train_score=False,
        )

        cv_df = pd.DataFrame(scores)

        # Make regression MAE positive for display
        if "test_neg_mae" in cv_df.columns:
            cv_df["test_mae"] = -cv_df["test_neg_mae"]

    return TrainResult(
        model_name=model_name,
        task=task,
        fit_time=fit_time,
        predict_time=predict_time,
        metrics=metrics,
        cv_metrics=cv_df,
    )


# -------------------------------------------------------------------------------------------------
# Sidebar: Upload + Core Controls
# -------------------------------------------------------------------------------------------------
st.sidebar.header("Dataset")

uploaded = st.sidebar.file_uploader(
    "Upload Account Balances (Excel) — File A",
    type=["xlsx", "xls"],
)

if not uploaded:
    st.info(
        "Upload an Excel export of File A (Account Balances). The README describes this as the required "
        "data source pulled from USASpending/GTAS. :contentReference[oaicite:1]{index=1}"
    )
    st.stop()

upload_bytes = uploaded.getvalue()
sheets = list_excel_sheets(upload_bytes)
sheet = st.sidebar.selectbox("Sheet", options=sheets, index=0)

df_raw = load_excel(upload_bytes, sheet_name=sheet)

st.sidebar.subheader("Cleaning")
drop_all_na = st.sidebar.checkbox("Drop rows with all-NA", value=True)
drop_dupes = st.sidebar.checkbox("Drop duplicates", value=True)

df = df_raw.copy()
if drop_all_na:
    df = df.dropna(how="all")
if drop_dupes:
    df = df.drop_duplicates()

st.sidebar.subheader("Columns")
numeric_cols_default = [c for c in DEFAULT_NUMERIC_COLUMNS if c in df.columns]
numeric_cols_inferred = _infer_numeric_columns(df)
numeric_cols = st.sidebar.multiselect(
    "Numeric columns (features/targets)",
    options=sorted(set(numeric_cols_inferred)),
    default=sorted(set(numeric_cols_default)) if numeric_cols_default else sorted(set(numeric_cols_inferred))[:10],
)

categorical_cols = _infer_categorical_columns(df, numeric_cols)

target_col = st.sidebar.selectbox(
    "Target column",
    options=[c for c in numeric_cols] if numeric_cols else list(df.columns),
    index=0,
)

feature_cols = st.sidebar.multiselect(
    "Feature columns",
    options=[c for c in df.columns if c != target_col],
    default=[c for c in numeric_cols if c != target_col],
)

st.sidebar.subheader("Split / CV")
test_size = st.sidebar.slider("Test size", min_value=0.1, max_value=0.5, value=0.25, step=0.05)
random_state = st.sidebar.number_input("Random state", min_value=0, max_value=10_000, value=42, step=1)
do_cv = st.sidebar.checkbox("Enable cross-validation", value=True)
cv_folds = st.sidebar.slider("CV folds", min_value=3, max_value=10, value=5, step=1)


# -------------------------------------------------------------------------------------------------
# Main Tabs
# -------------------------------------------------------------------------------------------------
tab_data, tab_desc, tab_inf, tab_feat, tab_models = st.tabs(
    ["Data", "Descriptive", "Inferential", "Feature Engineering", "Models"]
)


# -------------------------------------------------------------------------------------------------
# Data Tab
# -------------------------------------------------------------------------------------------------
with tab_data:
    st.subheader("Preview")
    st.write(f"Rows: {len(df):,} | Columns: {df.shape[1]:,}")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "null": df.isna().sum(),
            "unique": df.nunique(dropna=True),
        }
    ).sort_values(by="null", ascending=False)
    st.dataframe(summary, use_container_width=True)


# -------------------------------------------------------------------------------------------------
# Descriptive Statistics Tab
# -------------------------------------------------------------------------------------------------
with tab_desc:
    st.subheader("Descriptive Statistics (Numeric)")
    if not numeric_cols:
        st.warning("No numeric columns selected.")
    else:
        df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        st.dataframe(df_num.describe().T, use_container_width=True)

        st.subheader("Distributions")
        cols = st.multiselect("Select columns to plot", options=numeric_cols, default=numeric_cols[:3])
        bins = st.slider("Bins", min_value=10, max_value=200, value=40, step=5)

        if cols:
            for col in cols:
                fig = plt.figure(figsize=(8, 3))
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                plt.hist(s.values, bins=bins)
                plt.title(f"Histogram — {col}")
                plt.xlabel(col)
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)

        st.subheader("Boxplots")
        cols2 = st.multiselect("Select columns for boxplots", options=numeric_cols, default=numeric_cols[:5])
        if cols2:
            fig = plt.figure(figsize=(10, 3))
            data = [pd.to_numeric(df[c], errors="coerce").dropna().values for c in cols2]
            plt.boxplot(data, vert=False, labels=cols2)
            plt.title("Boxplots")
            st.pyplot(fig, clear_figure=True)


# -------------------------------------------------------------------------------------------------
# Inferential Statistics Tab
# -------------------------------------------------------------------------------------------------
with tab_inf:
    st.subheader("Correlation")
    if not numeric_cols:
        st.warning("No numeric columns selected.")
    else:
        df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

        corr_kind = st.selectbox("Correlation type", options=["pearson", "spearman"], index=0)
        corr = df_num.corr(method=corr_kind)

        st.dataframe(corr, use_container_width=True)

        st.subheader("Quick Hypothesis Tests")
        st.caption("Lightweight equivalents of what the notebook demonstrates (t-tests / ANOVA style checks).")

        col_a = st.selectbox("Column A", options=numeric_cols, index=0)
        col_b = st.selectbox("Column B", options=numeric_cols, index=min(1, len(numeric_cols) - 1))

        a = pd.to_numeric(df[col_a], errors="coerce").dropna()
        b = pd.to_numeric(df[col_b], errors="coerce").dropna()

        if len(a) > 2 and len(b) > 2:
            r, p = stats.pearsonr(a[: min(len(a), len(b))], b[: min(len(a), len(b))])
            st.write({"pearson_r": float(r), "p_value": float(p)})

        st.subheader("ANOVA (by a categorical column)")
        if categorical_cols:
            group_col = st.selectbox("Group by", options=categorical_cols, index=0)
            value_col = st.selectbox("Value column", options=numeric_cols, index=0)
            tmp = df[[group_col, value_col]].copy()
            tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
            tmp = tmp.dropna(subset=[group_col, value_col])

            groups = [g[value_col].values for _, g in tmp.groupby(group_col)]
            if len(groups) >= 2 and all(len(g) >= 2 for g in groups):
                fstat, pval = stats.f_oneway(*groups)
                st.write({"f_stat": float(fstat), "p_value": float(pval)})
            else:
                st.info("Need at least 2 groups with at least 2 observations each for ANOVA.")
        else:
            st.info("No categorical columns detected/selected for grouping.")


# -------------------------------------------------------------------------------------------------
# Feature Engineering Tab
# -------------------------------------------------------------------------------------------------
with tab_feat:
    st.subheader("Dimensionality Reduction")
    if not numeric_cols:
        st.warning("No numeric columns selected.")
    else:
        df_num = df[numeric_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        method = st.selectbox(
            "Method",
            options=["PCA", "IncrementalPCA", "TruncatedSVD", "FactorAnalysis"],
            index=0,
        )
        n_components = st.slider("Components", min_value=2, max_value=min(25, df_num.shape[1]), value=2, step=1)

        scaler_name = st.selectbox("Scaler", options=["standard", "robust", "none"], index=0)
        if scaler_name == "standard":
            scaled = StandardScaler().fit_transform(df_num.values)
        elif scaler_name == "robust":
            scaled = RobustScaler().fit_transform(df_num.values)
        else:
            scaled = df_num.values

        if method == "PCA":
            reducer = PCA(n_components=n_components, random_state=0)
        elif method == "IncrementalPCA":
            reducer = IncrementalPCA(n_components=n_components, batch_size=200)
        elif method == "TruncatedSVD":
            reducer = TruncatedSVD(n_components=n_components, random_state=0)
        else:
            reducer = FactorAnalysis(n_components=n_components, random_state=0)

        Xr = reducer.fit_transform(scaled)
        st.write("Reduced shape:", Xr.shape)

        if Xr.shape[1] >= 2:
            fig = plt.figure(figsize=(7, 4))
            plt.scatter(Xr[:, 0], Xr[:, 1], s=12)
            plt.title(f"{method} — Component 1 vs 2")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            st.pyplot(fig, clear_figure=True)

        if hasattr(reducer, "explained_variance_ratio_"):
            evr = getattr(reducer, "explained_variance_ratio_")
            if evr is not None:
                evr = np.asarray(evr)
                fig = plt.figure(figsize=(7, 3))
                plt.plot(np.arange(1, len(evr) + 1), evr, marker="o")
                plt.title("Explained Variance Ratio")
                plt.xlabel("Component")
                plt.ylabel("EVR")
                st.pyplot(fig, clear_figure=True)


# -------------------------------------------------------------------------------------------------
# Models Tab
# -------------------------------------------------------------------------------------------------
with tab_models:
    st.subheader("Unified Model Evaluation Pipeline")
    st.caption(
        "This tab implements the README’s idea of a single `train_and_evaluate()` interface that trains, "
        "optionally cross-validates, generates predictions, computes metrics, and exposes standard plots."
    )

    task = st.selectbox("Task", options=["regression", "classification", "clustering"], index=0)

    scaler = st.selectbox("Scaling", options=["standard", "robust", "none"], index=0)

    if not feature_cols:
        st.warning("Select at least one feature column in the sidebar.")
        st.stop()

    # Model registry (selected subset of what the notebook demonstrates)
    regression_models: Dict[str, Any] = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=random_state),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=None),
        "ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=300, random_state=random_state, n_jobs=None),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=random_state),
        "SGDRegressor": SGDRegressor(random_state=random_state),
        "KNeighborsRegressor": KNeighborsRegressor(n_neighbors=7),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(random_state=random_state, max_iter=1000),
    }

    classification_models: Dict[str, Any] = {
        "RidgeClassifier": RidgeClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=random_state),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=None),
        "ExtraTreesClassifier": ExtraTreesClassifier(n_estimators=300, random_state=random_state, n_jobs=None),
        "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(random_state=random_state),
        "SGDClassifier": SGDClassifier(random_state=random_state),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=7),
        "SVC": SVC(probability=True, random_state=random_state),
        "MLPClassifier": MLPClassifier(random_state=random_state, max_iter=1000),
        "GaussianNB": GaussianNB(),
        "BernoulliNB": BernoulliNB(),
    }

    clustering_models: Dict[str, Any] = {
        "KMeans": KMeans(n_clusters=3, random_state=random_state),
        "DBSCAN": DBSCAN(eps=0.5, min_samples=10),
    }

    if task in ("regression", "classification"):
        model_name = st.selectbox(
            "Model",
            options=sorted(regression_models.keys()) if task == "regression" else sorted(classification_models.keys()),
            index=0,
        )

        model = regression_models[model_name] if task == "regression" else classification_models[model_name]

        run = st.button("Train & Evaluate", type="primary")

        if run:
            with st.spinner("Training and evaluating..."):
                result = train_and_evaluate(
                    model=model,
                    model_name=model_name,
                    task=task,
                    df=df,
                    target_col=target_col,
                    feature_cols=feature_cols,
                    categorical_cols=categorical_cols,
                    numeric_cols=numeric_cols,
                    test_size=test_size,
                    random_state=random_state,
                    scaler=scaler,
                    do_cv=do_cv,
                    cv_folds=cv_folds,
                )

            left, right = st.columns(2)
            with left:
                st.markdown("#### Timings")
                st.write(
                    {
                        "fit_time": _fmt_seconds(result.fit_time),
                        "predict_time": _fmt_seconds(result.predict_time),
                    }
                )
            with right:
                st.markdown("#### Metrics")
                metrics_display = {k: v for k, v in result.metrics.items() if k != "Report"}
                st.write(metrics_display)

            if task == "classification":
                st.markdown("#### Classification Report")
                st.text(result.metrics.get("Report", ""))

            if result.cv_metrics is not None:
                st.markdown("#### Cross-Validation Summary")
                st.dataframe(result.cv_metrics.describe().T, use_container_width=True)

            # Standard plots mirroring README (“confusion matrix / ROC / PR / predicted vs actual / residuals”)
            st.markdown("#### Diagnostics")
            X_plot = df[feature_cols].copy()
            y_plot_raw = df[target_col].copy()

            # Rebuild a small train/test for diagnostics plots (consistent with evaluation above)
            if task == "classification":
                y_plot = _to_binary_by_median(y_plot_raw)
                strat = y_plot
            else:
                y_plot = pd.to_numeric(y_plot_raw, errors="coerce").fillna(0.0)
                strat = None

            X_train, X_test, y_train, y_test = train_test_split(
                X_plot,
                y_plot,
                test_size=test_size,
                random_state=random_state,
                stratify=strat,
            )

            # Recreate the pipeline to get access to predict_proba/decision_function consistently
            # (This is kept simple and deterministic rather than trying to extract the fitted pipe internals.)
            # Train a fresh pipeline for plots
            model2 = regression_models[model_name] if task == "regression" else classification_models[model_name]
            _ = train_and_evaluate(
                model=model2,
                model_name=model_name,
                task=task,
                df=df,
                target_col=target_col,
                feature_cols=feature_cols,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                test_size=test_size,
                random_state=random_state,
                scaler=scaler,
                do_cv=False,
                cv_folds=cv_folds,
            )

            # For plots, fit a dedicated pipeline we can use here
            # (duplicated logic from train_and_evaluate but kept explicit for clarity)
            if scaler == "standard":
                num_scaler = StandardScaler()
            elif scaler == "robust":
                num_scaler = RobustScaler()
            else:
                num_scaler = None

            numeric_transform = [("imputer", SimpleImputer(strategy="median"))]
            if num_scaler is not None:
                numeric_transform.append(("scaler", num_scaler))

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        Pipeline(steps=numeric_transform),
                        [c for c in numeric_cols if c in feature_cols],
                    ),
                    (
                        "cat",
                        Pipeline(
                            steps=[
                                ("imputer", SimpleImputer(strategy="most_frequent")),
                                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                            ]
                        ),
                        [c for c in categorical_cols if c in feature_cols],
                    ),
                ],
                remainder="drop",
            )

            pipe = Pipeline(steps=[("prep", preprocessor), ("model", model2)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            if task == "regression":
                # Predicted vs Actual
                fig = plt.figure(figsize=(6, 4))
                plt.scatter(y_test, y_pred, s=12)
                plt.title("Actual vs Predicted")
                plt.xlabel("Actual")
                plt.ylabel("Predicted")
                st.pyplot(fig, clear_figure=True)

                # Residuals
                resid = y_test - y_pred
                fig = plt.figure(figsize=(6, 3))
                plt.hist(resid, bins=40)
                plt.title("Residual Distribution")
                plt.xlabel("Residual")
                plt.ylabel("Count")
                st.pyplot(fig, clear_figure=True)

            else:
                # Confusion matrix
                fig = plt.figure(figsize=(5, 4))
                ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
                plt.title("Confusion Matrix")
                st.pyplot(fig, clear_figure=True)

                # ROC + PR (if probabilities exist)
                proba = None
                if hasattr(pipe, "predict_proba"):
                    try:
                        proba = pipe.predict_proba(X_test)[:, 1]
                    except Exception:
                        proba = None

                if proba is not None:
                    fig = plt.figure(figsize=(5, 4))
                    RocCurveDisplay.from_predictions(y_test, proba)
                    plt.title("ROC Curve")
                    st.pyplot(fig, clear_figure=True)

                    fig = plt.figure(figsize=(5, 4))
                    PrecisionRecallDisplay.from_predictions(y_test, proba)
                    plt.title("Precision-Recall Curve")
                    st.pyplot(fig, clear_figure=True)

    else:
        st.markdown("#### Clustering")
        model_name = st.selectbox("Clusterer", options=sorted(clustering_models.keys()), index=0)
        clusterer = clustering_models[model_name]

        # Use only numeric features for clustering (typical)
        num_features_for_cluster = [c for c in feature_cols if c in numeric_cols]
        if len(num_features_for_cluster) < 2:
            st.warning("For clustering, select at least 2 numeric feature columns.")
            st.stop()

        X = df[num_features_for_cluster].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
        X = StandardScaler().fit_transform(X)

        run = st.button("Fit Clustering", type="primary")
        if run:
            with st.spinner("Clustering..."):
                t0 = time.perf_counter()
                labels = clusterer.fit_predict(X)
                t1 = time.perf_counter()

            st.write(
                {
                    "model": model_name,
                    "fit_predict_time": _fmt_seconds(t1 - t0),
                    "clusters_found": int(len(set(labels)) - (1 if -1 in set(labels) else 0)),
                    "noise_points": int(np.sum(labels == -1)) if -1 in set(labels) else 0,
                }
            )

            if X.shape[1] >= 2:
                fig = plt.figure(figsize=(6, 4))
                plt.scatter(X[:, 0], X[:, 1], c=labels, s=12)
                plt.title(f"{model_name} — Cluster Map (first 2 standardized features)")
                plt.xlabel(num_features_for_cluster[0])
                plt.ylabel(num_features_for_cluster[1])
                st.pyplot(fig, clear_figure=True)

