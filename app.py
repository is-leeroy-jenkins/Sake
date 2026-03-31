from __future__ import annotations

import config as cfg
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from scipy.stats.mstats import winsorize

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder,
    MinMaxScaler,
    RobustScaler,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import (
    IsolationForest,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LogisticRegression,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
)

# -----------------------------------------------------------------------------
# Config Initialization
# -----------------------------------------------------------------------------
st.logo( cfg.LOGO, size='large' )

st.set_page_config( page_title='Sake', layout='wide', page_icon=cfg.FAVICON,
    initial_sidebar_state='expanded', )

sns.set_theme(style='darkgrid')

def styled_scatter( ax: plt.Axes, x: np.ndarray, y: np.ndarray,  series_index: int = 0,
    label: Optional[str] = None,  size: int = 30, ) -> None:
    """
    
	    Purpose:
	    ________
	    Draw a consistently styled scatter plot with clear point boundaries and
	    visually distinct series.
	
	    Parameters:
	    ___________
	    ax : plt.Axes
	        Matplotlib axes to draw on.
	    x : np.ndarray
	        X-coordinates of the points.
	    y : np.ndarray
	        Y-coordinates of the points.
	    series_index : int, optional
	        Index used to pick color and marker from predefined palettes.
	    label : Optional[str], optional
	        Legend label for the series, if any.
	    size : int, optional
	        Marker size for the scatter plot.
	
	    Returns:
	    ________
	    None
	        This function draws on the provided axes in-place.
        
    """
    color = cfg.PALETTE[series_index % len(cfg.PALETTE)]
    marker = cfg.MARKERS[series_index % len(cfg.MARKERS)]
    ax.scatter( x,  y, s=size, alpha=0.9,  edgecolors="#020617",
        linewidths=0.6,  c=[color],  marker=marker,  label=label,  )
    ax.grid(True, alpha=0.25)


def auto_float_format(series: pd.Series, max_decimals: int = 4) -> str:
    """
    
	    Purpose:
	    ________
	    Infer a reasonable float formatting pattern for a numeric series based on
	    its scale, so large values are readable and decimals are not excessive.
	
	    Parameters:
	    ___________
	    series : pd.Series
	        Series whose numeric magnitude is used to pick the format.
	    max_decimals : int, optional
	        Maximum number of decimal places allowed in the format string.
	
	    Returns:
	    ________
	    str
	        A Python format string such as '{:,.2f}' appropriate for the series.
	        
    """
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return "{:,.2f}"

    mag = float(np.nanpercentile(np.abs(s.values), 95))

    if mag >= 1e9:
        decimals = 0
    elif mag >= 1e6:
        decimals = 1
    elif mag >= 1e3:
        decimals = 2
    elif mag >= 1:
        decimals = 3
    else:
        decimals = 4

    decimals = min(decimals, max_decimals)
    return f"{{:,.{decimals}f}}"


def render_table( df: pd.DataFrame,  title: Optional[str] = None,  caption: Optional[str] = None,
    precision: int = 4, dark_mode: bool = True, max_rows: int = 500, ) -> None:
    """
    
	    Purpose:
	    ________
	    Render a pandas DataFrame as a styled HTML table suitable for Streamlit,
	    avoiding raw JSON output.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        DataFrame to render.
	    title : Optional[str], optional
	        Optional section title to display above the table.
	    caption : Optional[str], optional
	        Descriptive caption explaining the table content.
	    precision : int, optional
	        Maximum decimal precision for numeric columns.
	    dark_mode : bool, optional
	        If True, use dark-themed table colors; otherwise light theme.
	    max_rows : int, optional
	        Maximum number of rows to display; additional rows are truncated.
	
	    Returns:
	    ________
	    None
	        The function writes directly to the active Streamlit app.
	        
    """
    if title:
        st.markdown(f'#### {title}')

    if df is None or df.empty:
        st.info("No data to display.")
        return

    df_show = df.copy()
    if len(df_show) > max_rows:
        df_show = df_show.head(max_rows)

    num_cols = df_show.select_dtypes(include=[np.number]).columns.tolist()
    fmt = {c: auto_float_format(df_show[c], precision) for c in num_cols}

    if dark_mode:
        text = '#F9FAFB'
        header_bg = '#1F2937'
        row_even = '#020617'
        row_odd = '#030712'
        border = '#374151'
    else:
        text = '#111827'
        header_bg = '#E5E7EB'
        row_even = '#FFFFFF'
        row_odd = '#F9FAFB'
        border = '#D1D5DB'

    styler = (
        df_show.style
        .format(fmt)
        .set_table_styles(
            [
                {
                    'selector': 'table',
                    'props': [
                        ('border-collapse', 'collapse'),
                        ('width', '100%'),
                        ('font-size', '0.85rem'),
                    ],
                },
                {
                    'selector': 'th',
                    'props': [
                        ('background-color', header_bg),
                        ('color', text),
                        ('border', f'1px solid {border}'),
                        ('padding', '6px 8px'),
                        ('font-weight', '600'),
                        ('text-align', 'left'),
                        ('white-space', 'nowrap'),
                    ],
                },
                {
                    'selector': 'td',
                    'props': [
                        ('color', text),
                        ('border', f'1px solid {border}'),
                        ('padding', '4px 8px'),
                        ('white-space', 'nowrap'),
                    ],
                },
                {
                    'selector': 'tr:nth-child(even) td',
                    'props': [('background-color', row_even)],
                },
                {
                    'selector': 'tr:nth-child(odd) td',
                    'props': [('background-color', row_odd)],
                },
            ]
        )
    )
    st.markdown(styler.to_html(), unsafe_allow_html=True)

    if caption:
        st.caption(caption)

def safe_numeric_series(df: pd.DataFrame, col: str) -> np.ndarray:
    """
    
	    Purpose:
	    ________
	    Convert a DataFrame column to a clean numeric NumPy array, dropping any
	    non-numeric or missing values.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        Source DataFrame containing the column.
	    col : str
	        Name of the column to convert.
	
	    Returns:
	    ________
	    np.ndarray
	        One-dimensional array of float values with NaNs removed.
	        
    """
    v = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(float)
    return v

def descriptive_profile(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """
    
	    Purpose:
	    ________
	    Compute an extended descriptive statistics profile for a set of numeric
	    columns, including tails, dispersion measures, and simple outlier rates.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        DataFrame containing the numeric columns.
	    cols : List[str]
	        List of column names to profile.
	
	    Returns:
	    ________
	    pd.DataFrame
	        DataFrame with one row per feature and many descriptive statistics
	        columns (mean, std, quantiles, skew, kurtosis, outlier rates, etc.).
	        
    """
    rows: List[Dict[str, Any]] = []
    n = df.shape[0]
    percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

    for c in cols:
        v = safe_numeric_series(df, c)
        non_missing = int(np.isfinite(v).sum())
        missing = int(n - non_missing)
        if non_missing == 0:
            continue

        q_vals = np.nanpercentile(v, percentiles)
        q = dict(zip(percentiles, q_vals))

        mean = float(np.nanmean(v))
        std = float(np.nanstd(v, ddof=0))
        var = float(np.nanvar(v, ddof=0))
        med = float(np.nanmedian(v))
        mad = float(np.nanmedian(np.abs(v - med)))
        iqr = float(q[75] - q[25])
        rng = float(q[100] - q[0])

        skew = float(stats.skew(v)) if v.size >= 3 else 0.0
        kurt = float(stats.kurtosis(v)) if v.size >= 4 else 0.0
        zero_pct = float((v == 0).mean() * 100.0)

        lo = q[25] - 1.5 * iqr
        hi = q[75] + 1.5 * iqr
        out_iqr = float(((v < lo) | (v > hi)).mean() * 100.0)

        z = (v - mean) / (std + 1e-12)
        out_z3 = float((np.abs(z) > 3.0).mean() * 100.0)

        normal_p: Optional[float] = None
        try:
            if 8 <= v.size <= 5000:
                _, p = stats.shapiro(v)
                normal_p = float(p)
            elif v.size > 5000:
                _, p = stats.normaltest(v[:5000])
                normal_p = float(p)
            elif v.size >= 8:
                _, p = stats.normaltest(v)
                normal_p = float(p)
        except Exception:
            normal_p = None

        rows.append(
            {
                "feature": c,
                "count": int(v.size),
                "missing_pct": float((missing / n) * 100.0) if n else 0.0,
                "mean": mean,
                "std": std,
                "var": var,
                "min": float(q[0]),
                "p01": float(q[1]),
                "p05": float(q[5]),
                "p10": float(q[10]),
                "q1": float(q[25]),
                "median": float(q[50]),
                "q3": float(q[75]),
                "p90": float(q[90]),
                "p95": float(q[95]),
                "p99": float(q[99]),
                "max": float(q[100]),
                "iqr": iqr,
                "range": rng,
                "mad": mad,
                "skew": skew,
                "kurtosis": kurt,
                "zero_pct": zero_pct,
                "outlier_iqr_pct": out_iqr,
                "outlier_z3_pct": out_z3,
                "normality_p": normal_p,
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["missing_pct", "outlier_iqr_pct"], ascending=[True, False]
    )

def feature_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    
	    Purpose:
	    ________
	    Summarize data quality signals for each column, such as completeness,
	    uniqueness, and simple variance/entropy indicators.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        DataFrame whose columns are to be evaluated.
	
	    Returns:
	    ________
	    pd.DataFrame
	        DataFrame with one row per column, including completeness percentage,
	        unique counts, cardinality ratio, and variance/entropy metrics.
        
    """
    rows: List[Dict[str, Any]] = []
    n = df.shape[0]
    for c in df.columns:
        s = df[c]
        non_missing = int(s.notna().sum())
        completeness = float((non_missing / n) * 100.0) if n else 0.0
        uniq = int(s.nunique(dropna=True))
        card_ratio = float(uniq / non_missing) if non_missing else 0.0

        variance = np.nan
        entropy = np.nan

        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors="coerce").dropna().values.astype(float)
            variance = float(np.var(v)) if v.size else np.nan
        else:
            vc = s.dropna().astype(str).value_counts(normalize=True)
            entropy = float(stats.entropy(vc.values)) if vc.size > 1 else 0.0

        rows.append(
            {
                "feature": c,
                "dtype": str(s.dtype),
                "completeness_pct": completeness,
                "unique_values": uniq,
                "cardinality_ratio": card_ratio,
                "variance": variance,
                "entropy": entropy,
            }
        )

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["completeness_pct", "cardinality_ratio"], ascending=[False, False]
    )

def corr_with_pvalues( df: pd.DataFrame, cols: List[str], 
		method: str ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    
	    Purpose:
	    ________
	    Compute a correlation matrix and the corresponding matrix of p-values for
	    a set of numeric columns using Pearson or Spearman correlation.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        DataFrame containing the columns of interest.
	    cols : List[str]
	        List of column names to include in the correlation computation.
	    method : str
	        Correlation type; either 'pearson' or 'spearman'.
	
	    Returns:
	    ________
	    Tuple[pd.DataFrame, pd.DataFrame]
	        Two DataFrames with identical indices/columns:
	        - correlation coefficients
	        - p-values for the corresponding tests.
        
    """
    corr = pd.DataFrame(index=cols, columns=cols, dtype=float)
    pval = pd.DataFrame(index=cols, columns=cols, dtype=float)

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if j < i:
                corr.loc[a, b] = corr.loc[b, a]
                pval.loc[a, b] = pval.loc[b, a]
                continue

            x = pd.to_numeric(df[a], errors="coerce")
            y = pd.to_numeric(df[b], errors="coerce")
            m = x.notna() & y.notna()

            if int(m.sum()) < 3:
                r, p = np.nan, np.nan
            else:
                if method == "pearson":
                    r, p = stats.pearsonr(x[m].values, y[m].values)
                else:
                    r, p = stats.spearmanr(x[m].values, y[m].values)

            corr.loc[a, b] = float(r)
            pval.loc[a, b] = float(p)

    return corr, pval

def numeric_default_candidates( df: pd.DataFrame, numeric_cols: List[str] ) -> List[str]:
    """
	    
	    Purpose:
	    ________
	    Identify numeric columns that are likely to be measures (not IDs or codes)
	    and should be pre-selected in numeric analyses.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        Source DataFrame containing the numeric columns.
	    numeric_cols : List[str]
	        List of all numeric column names inferred from dtypes.
	
	    Returns:
	    ________
	    List[str]
	        Subset of numeric column names to use as sensible defaults.
	        
    """
    defaults: List[str] = []
    for col in numeric_cols:
        name = col.lower()

        s_num = pd.to_numeric(df[col], errors="coerce")
        v = s_num.dropna().values
        if v.size == 0:
            continue

        # Exclude pure integer-valued columns from defaults.
        if np.all(np.isfinite(v)) and np.all(np.floor(v) == v):
            continue

        # Skip obvious IDs/codes based on name.
        if name.endswith("id") or name.endswith("_id") or "code" in name:
            continue

        # High-cardinality near-unique numeric looks like row IDs.
        non_missing = int(np.isfinite(v).sum())
        if non_missing == 0:
            continue
        uniq = len(np.unique(v))
        ratio = uniq / max(non_missing, 1)

        if non_missing >= 50 and ratio > 0.98:
            continue

        defaults.append(col)

    if not defaults:
        defaults = numeric_cols[:]
    return defaults

def categorical_default_candidates( df: pd.DataFrame, cat_cols: List[str] ) -> List[str]:
    """
    
	    Purpose:
	    ________
	    Identify categorical columns that behave like true categories (not
	    near-unique IDs) and are good candidates for pre-selection.
	
	    Parameters:
	    ___________
	    df : pd.DataFrame
	        Source DataFrame containing the categorical columns.
	    cat_cols : List[str]
	        List of column names treated as categorical.
	
	    Returns:
	    ________
	    List[str]
	        Subset of categorical column names that have reasonable cardinality
	        and should be pre-selected by default.
	        
    """
    defaults: List[str] = []
    for col in cat_cols:
        s = df[col].astype(str)
        non_missing = int(s.notna().sum())
        if non_missing == 0:
            continue
        uniq = int(s.nunique(dropna=True))
        ratio = uniq / max(non_missing, 1)

        if ratio <= 0.5:
            defaults.append(col)

    if not defaults:
        defaults = cat_cols[:]
    return defaults

def style_subheaders( ) -> None:
	"""
	
		Purpose:
		_________
		Sets the style of subheaders in the main UI
		
	"""
	st.markdown(
		"""
		<style>
		div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stMarkdownContainer"] h3,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h2,
		div[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] h3 {
			color: rgb(0, 120, 252) !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	
# -----------------------------------------------------------------------------
# Data loading  
# -----------------------------------------------------------------------------
style_subheaders( )
st.sidebar.header( '📁 Data Input' )
use_fallback = st.sidebar.checkbox( 'Load fallback data', value=True, key='use_fallback', )
uploaded = st.sidebar.file_uploader( 'Upload File', type=['xlsx', 'xls'], key='upload_file', )

df: Optional[pd.DataFrame] = None
data_source: str = ""

if uploaded is not None:
	xls = pd.ExcelFile( uploaded )
	sheet = st.sidebar.selectbox( 'Sheet', options=xls.sheet_names, index=0, key='sheet_upload', )
	df = xls.parse( sheet )
	data_source = f'Uploaded: {uploaded.name} / {sheet}'
elif use_fallback:
	if not cfg.FALLBACK_PATH.exists( ):
		st.error( f'Fallback file not found: {cfg.FALLBACK_PATH}' )
		st.stop( )
	xls = pd.ExcelFile( cfg.FALLBACK_PATH )
	sheet = st.sidebar.selectbox(
		'Sheet',
		options=xls.sheet_names,
		index=0,
		key='sheet_fallback',
	)
	df = xls.parse( sheet )
	data_source = f'Fallback: {cfg.FALLBACK_PATH.name} / {sheet}'

if df is None or df.empty:
	st.info( 'Please upload a dataset or enable the fallback sample.' )
	st.stop( )

st.sidebar.caption( f'📄 Source: {data_source}' )

# -----------------------------------------------------------------------------
# Column registry + dtype-aware defaults
# -----------------------------------------------------------------------------
numeric_raw: List[ str ] = df.select_dtypes( include=[ np.number ] ).columns.tolist( )
categorical_raw: List[ str ] = [ c for c in df.columns if c not in numeric_raw ]

numeric_default: List[ str ] = numeric_default_candidates( df, numeric_raw )
categorical_default: List[ str ] = categorical_default_candidates( df, categorical_raw )

COLUMN_REGISTRY: Dict[ str, List[ str ] ] = \
{
	'numeric_raw': numeric_raw,
	'categorical_raw': categorical_raw,
	'numeric_default': numeric_default,
	'categorical_default': categorical_default,
}

numeric = numeric_raw
categorical = categorical_raw

# -----------------------------------------------------------------------------
# Global display controls
# -----------------------------------------------------------------------------	
st.sidebar.header( '⚙️ Display Controls' )
preview_rows = st.sidebar.slider( 'Preview rows', 10, 300, 50, 10, key='preview_rows' )
dark_tables = st.sidebar.checkbox( 'Use dark tables', value=True, key='dark_tables' )
plot_theme = st.sidebar.selectbox( 'Plot theme', [ 'Dark', 'Light' ], index=0, key='plot_theme' )

if plot_theme == 'Dark':
	plt.style.use( 'dark_background' )
	sns.set_theme( style='darkgrid' )
else:
	plt.style.use( 'default' )
	sns.set_theme( style='whitegrid' )

# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------
st.markdown( '## Sake' )
st.caption( 'Account Balances' )
tabs = st.tabs( cfg.TABS )

# =============================================================================
# 1. Data Overview
# =============================================================================
with tabs[ 0 ]:
	c1, c2, c3, c4 = st.columns( 4, border=True )
	c1.metric( 'Rows', f'{df.shape[ 0 ]:,}' )
	c2.metric( 'Columns', f'{df.shape[ 1 ]:,}' )
	c3.metric( 'Numeric', len( numeric ) )
	c4.metric( 'Categorical', len( categorical ) )
	
	st.divider( )
	st.subheader( 'Preview' )
	st.dataframe( df.head( preview_rows ), use_container_width=True, height=420 )
	st.divider( )
	st.subheader( 'Feature Quality' )
	fq = feature_quality( df )
	st.data_editor( fq )
	
	st.divider( )
	st.subheader( 'Schema Summary' )
	schema = pd.DataFrame(
		{
				'column': df.columns,
				'dtype': [ str( df[ c ].dtype ) for c in df.columns ],
				'non_missing': [ int( df[ c ].notna( ).sum( ) ) for c in df.columns ],
				'missing': [ int( df[ c ].isna( ).sum( ) ) for c in df.columns ],
				'unique': [ int( df[ c ].nunique( dropna=True ) ) for c in df.columns ],
		}
	).sort_values( [ 'dtype', 'missing', 'unique' ], ascending=[ True, True, False ] )
	
	st.data_editor( schema )

# =============================================================================
# Descriptive Statistics
# =============================================================================
with tabs[ 1 ]:
	if not numeric:
		st.warning( 'No numeric columns detected.' )
	else:
		st.divider( )
		st.subheader( 'Numeric Profile' )
		
		prof = descriptive_profile( df, numeric )
		st.data_editor( prof, )
		
		st.subheader( 'Distributions by Feature' )
		default_hist = numeric_default[ : min( 6, len( numeric_default ) ) ]
		num_sel = st.multiselect( 'Numeric columns for histograms/boxplots', numeric,
			default=default_hist, key="desc_num_sel", )
		bins = st.slider( "Histogram bins", 10, 100, 40, 5, key="desc_bins" )
		
		for i, col in enumerate( num_sel ):
			v = safe_numeric_series( df, col )
			if v.size < 2:
				continue
			
			color = cfg.PALETTE[ i % len( cfg.PALETTE ) ]
			
			fig, ax = plt.subplots( figsize=(8, 3) )
			ax.hist( v, bins=bins, alpha=0.85, edgecolor='#0f172a', linewidth=0.6, color=color, )
			ax.set_title( col )
			ax.set_ylabel( 'Count' )
			ax.grid( True, alpha=0.25 )
			st.pyplot( fig )
			st.caption( f'Histogram of {col}: look for heavy tails, multiple modes, '
				'or spikes at specific values.' )
			
			fig2, ax2 = plt.subplots( figsize=(6, 3) )
			bp = ax2.boxplot( v, vert=False, showfliers=True, patch_artist=True,
				boxprops={ 'facecolor': color, 'edgecolor': '#020617', 'linewidth': 0.8, },
				medianprops={ 'color': '#f9fafb', 'linewidth': 1.4 },
				whiskerprops={ 'color': '#9ca3af', 'linewidth': 0.8 },
				capprops={ 'color': '#9ca3af', 'linewidth': 0.8 },
				flierprops={ 'marker': 'o', 'markersize': 3, 'markerfacecolor': color }, )
			
			for patch in bp[ 'boxes' ]:
				patch.set_facecolor( color )
				patch.set_alpha( 0.85 )
			
			ax2.set_title( f'Boxplot: {col}' )
			ax2.set_xlabel( 'Value' )
			ax2.grid( True, axis='x', alpha=0.25 )
			st.pyplot( fig2 )
			st.caption( f'Boxplot of {col}: outliers beyond whiskers may represent '
				'data errors or genuinely extreme observations.' )
		
		st.subheader( 'Normality Diagnostics (Q–Q Plots)' )
		default_qq = numeric_default[ : min( 3, len( numeric_default ) ) ]
		qq_sel = st.multiselect( 'Numeric columns for Q–Q plots', numeric, default=default_qq,
			key='desc_qq_sel', )
		for i, col in enumerate( qq_sel ):
			v = safe_numeric_series( df, col )
			if v.size < 20:
				continue
			
			# Use styled_scatter for Q-Q points and explicit reference line.
			(osm, osr), (slope, intercept, r) = stats.probplot( v, dist="norm" )
			fig, ax = plt.subplots( figsize=(6, 4) )
			styled_scatter( ax, np.asarray( osm ), np.asarray( osr ), series_index=i, size=30, )
			x_fit = np.array( [ np.min( osm ), np.max( osm ) ] )
			y_fit = slope * x_fit + intercept
			ax.plot( x_fit, y_fit, linestyle="--", linewidth=1.2, color="#f97316",
				label=f"Normal fit (r={r:.3f})", )
			
			ax.set_title( f"Q–Q Plot: {col}" )
			ax.set_xlabel( "Theoretical quantiles" )
			ax.set_ylabel( "Ordered sample values" )
			ax.legend( fontsize=8 )
			st.pyplot( fig )
			st.caption( f"For {col}, points hugging the dashed line suggest approximate normality; "
				"systematic curvature or heavy tails indicate deviation." )

# =============================================================================
# Inferential Statistics
# =============================================================================
with tabs[ 2 ]:
	if not numeric:
		st.warning( 'Inferential statistics require numeric measures.' )
	else:
		st.subheader( 'Correlation Analysis' )
		default_corr = numeric_default[ : min( 10, len( numeric_default ) ) ]
		corr_cols = st.multiselect( 'Numeric columns for correlation', numeric, default=default_corr,
			key='inf_corr_cols', )
		corr_method = st.selectbox( 'Correlation method', [ 'pearson', 'spearman' ], index=0,
			key="inf_corr_method", )
		
		if len( corr_cols ) >= 2:
			corr_mat, p_mat = corr_with_pvalues( df, corr_cols, corr_method )
			
			fig, ax = plt.subplots( figsize=(8, 6) )
			sns.heatmap( corr_mat, cmap="coolwarm", center=0, annot=True, fmt=".2f",
				linewidths=0.5, linecolor='black', square=True, vmin=-1.0, vmax=1.0,
				cbar_kws={ 'shrink': 0.8 }, annot_kws={ 'size': 7 }, ax=ax, )
			ax.set_title( f'{corr_method.title( )} Correlation Heatmap' )
			st.pyplot( fig )
			st.caption( 'Warm cells indicate positive association; cool cells indicate negative '
				'association. '
				'Magnitudes near ±1 suggest strong linear/monotonic relationships.' )
			
			render_table( corr_mat.reset_index( ).rename( columns={ 'index': 'feature' } ),
				title='Correlation matrix', dark_mode=dark_tables, precision=4 )
			render_table( p_mat.reset_index( ).rename( columns={ 'index': 'feature' } ),
				title='Correlation p-values', dark_mode=dark_tables, precision=6, )
		
		st.subheader( 'Confidence Intervals for Means' )
		default_ci = numeric_default[ : min( 8, len( numeric_default ) ) ]
		ci_cols = st.multiselect( 'Numeric columns for CI', numeric, default=default_ci,
			key='inf_ci_cols', )
		conf = st.slider( 'Confidence level', 0.80, 0.99, 0.95, 0.01, key='inf_ci_conf' )
		ci_rows: List[ Dict[ str, Any ] ] = [ ]
		for col in ci_cols:
			v = safe_numeric_series( df, col )
			if v.size < 2:
				continue
			mean = float( v.mean( ) )
			se = float( stats.sem( v ) )
			dfree = int( v.size - 1 )
			tcrit = float( stats.t.ppf( (1.0 + conf) / 2.0, dfree ) )
			lo = mean - tcrit * se
			hi = mean + tcrit * se
			ci_rows.append(
				{
						'feature': col,
						'n': int( v.size ),
						'mean': mean,
						'ci_low': lo,
						'ci_high': hi,
				}
			)
		
		render_table( pd.DataFrame( ci_rows ),
			caption='Confidence intervals quantify uncertainty around each mean estimate.',
			dark_mode=dark_tables, precision=4, )
		
		st.subheader( 'Two-Group Comparisons' )
		if categorical:
			num_feature = st.selectbox( 'Numeric feature', numeric,
				index=( numeric.index( numeric_default[ 0 ] )
						if numeric_default and numeric_default[ 0 ] in numeric
						else 0 ), key='inf_num_feature', )
			group_col = st.selectbox( 'Grouping (categorical)', categorical, index=0,
				key='inf_group_col', )
			group_vals = ( df[ group_col ].dropna( ).astype( str ).unique( ).tolist( ) )
			if len( group_vals ) >= 2:
				sel_groups = st.multiselect( 'Select exactly two groups', options=group_vals,
					default=group_vals[ :2 ], key='inf_sel_groups', )
				test_kind = st.selectbox( 'Test type', [ 'Two-sample t-test', 'Mann–Whitney U' ],
					index=0, key='inf_test_kind', )
				
				if len( sel_groups ) == 2:
					a = pd.to_numeric( df.loc[ df[ group_col ].astype( str ) == sel_groups[ 0 ], num_feature ],
						errors='coerce', ).dropna( )
					b = pd.to_numeric( df.loc[ df[ group_col ].astype( str ) == sel_groups[ 1 ], num_feature ],
						errors='coerce', ).dropna( )
					
					if a.size >= 2 and b.size >= 2:
						rows: List[ Dict[ str, Any ] ] = [ ]
						if test_kind == 'Two-sample t-test':
							t_stat, p_val = stats.ttest_ind( a, b, equal_var=False )
							rows.append(
								{
										'test': 't-test (Welch)',
										'group_A': sel_groups[ 0 ],
										'group_B': sel_groups[ 1 ],
										'n_A': int( a.size ),
										'n_B': int( b.size ),
										'mean_A': float( a.mean( ) ),
										'mean_B': float( b.mean( ) ),
										't': float( t_stat ),
										'p_value': float( p_val ),
								}
							)
							caption = ( 'Welchs t-test compares means without assuming equal '
									'variances.' )
						else:
							u_stat, p_val = stats.mannwhitneyu( a, b, alternative='two-sided' )
							rows.append(
								{
										'test': 'Mann–Whitney U',
										'group_A': sel_groups[ 0 ],
										'group_B': sel_groups[ 1 ],
										'n_A': int( a.size ),
										'n_B': int( b.size ),
										'median_A': float( np.median( a ) ),
										'median_B': float( np.median( b ) ),
										'U': float( u_stat ),
										'p_value': float( p_val ),
								} )
							
							caption = ( 'Mann–Whitney compares distributions without assuming '
									'normality.' )
						
						render_table(
							pd.DataFrame( rows ),
							caption=caption,
							dark_mode=dark_tables,
							precision=6,
						)
						
						fig, ax = plt.subplots( figsize=(8, 3) )
						color_a = cfg.PALETTE[ 0 ]
						color_b = cfg.PALETTE[ 1 ]
						ax.hist( a, bins=30, alpha=0.6, edgecolor='#020617', linewidth=0.5,
							label=sel_groups[ 0 ], color=color_a, )
						ax.hist( b, bins=30, alpha=0.6, edgecolor='#020617',
							linewidth=0.5, label=sel_groups[ 1 ], color=color_b, )
						ax.set_title( f'{num_feature} by {group_col}' )
						ax.legend( )
						ax.grid( True, alpha=0.25 )
						st.pyplot( fig )
						st.caption(
							'Overlapping histograms show practical differences between groups '
							'beyond the p-value.'
						)

# =============================================================================
# Feature Analysis
# =============================================================================
with tabs[ 3 ]:
	if len( numeric ) < 2:
		st.warning( 'Feature analysis requires at least two numeric columns.' )
	else:
		default_fa = numeric_default[ : min( 12, len( numeric_default ) ) ]
		fa_cols = st.multiselect( 'Numeric features for analysis', numeric, default=default_fa,
			key='fa_cols', )
		
		if len( fa_cols ) >= 2:
			X = df[ fa_cols ].apply( pd.to_numeric, errors='coerce' ).dropna( )
			st.divider( )
			st.subheader( 'Correlation Heatmap' )
			corr = X.corr( )
			fig, ax = plt.subplots( figsize=(8, 6) )
			sns.heatmap( corr, cmap='coolwarm', center=0, annot=True, fmt='.2f',
				linewidths=0.5, linecolor='black', square=True, vmin=-1.0, vmax=1.0,
				cbar_kws={ 'shrink': 0.8 }, annot_kws={ 'size': 7 }, ax=ax, )
			ax.set_title( 'Correlation Heatmap' )
			st.pyplot( fig )
			st.caption(
				'Highly correlated features may be redundant; consider dropping or combining them.'
			)
			
			pairs: List[ Dict[ str, Any ] ] = [ ]
			for i in range( len( fa_cols ) ):
				for j in range( i + 1, len( fa_cols ) ):
					a, b = fa_cols[ i ], fa_cols[ j ]
					r = float( corr.loc[ a, b ] )
					pairs.append(
						{
								'feature_A': a,
								'feature_B': b,
								'corr': r,
								'abs_corr': abs( r ),
						}
					)
			top_pairs = (
					pd.DataFrame( pairs )
					.sort_values( 'abs_corr', ascending=False )
					.head( 20 )
					.drop( columns=[ 'abs_corr' ] )
			)
			render_table( top_pairs, title='Top correlated feature pairs',
				dark_mode=dark_tables, precision=4, )
			
			st.subheader( 'PCA (Principal Component Analysis)' )
			X_scaled = StandardScaler( ).fit_transform( X.values )
			n_comp = st.slider(
				'Number of components', 2, min( 10, len( fa_cols ) ), 3, 1, key='fa_pca_n'
			)
			pca = PCA( n_components=n_comp, random_state=42 )
			Z = pca.fit_transform( X_scaled )
			
			evr = pd.DataFrame(
				{
						'component': [ f'PC{i + 1}' for i in
						               range( len( pca.explained_variance_ratio_ ) ) ],
						'explained_variance_ratio': pca.explained_variance_ratio_,
						'cumulative': np.cumsum( pca.explained_variance_ratio_ ),
				}
			)
			render_table( evr,
				caption='Cumulative variance shows how many components are needed to capture most '
				        'signal.',
				dark_mode=dark_tables, precision=4, )
			
			if Z.shape[ 1 ] >= 2:
				fig3, ax3 = plt.subplots( figsize=(8, 6) )
				styled_scatter( ax3, Z[ :, 0 ], Z[ :, 1 ], series_index=0, size=30 )
				ax3.set_xlabel( 'PC1' )
				ax3.set_ylabel( 'PC2' )
				ax3.set_title( 'PCA Projection (PC1 vs PC2)' )
				st.pyplot( fig3 )
				st.caption( 'Separated clusters in PCA space suggest meaningful latent groupings.' )
			
			st.subheader( 'k-Means Clustering (PCA space)' )
			k = st.slider( 'Number of clusters (k)', 2, 12, 3, 1, key='fa_k' )
			km = KMeans( n_clusters=k, n_init=10, random_state=42 )
			labels = km.fit_predict( Z )
			
			counts = (
					pd.Series( labels )
					.value_counts( )
					.sort_index( )
					.rename_axis( 'cluster' )
					.reset_index( name='count' )
			)
			render_table( counts,
				caption='Cluster counts show the size of each k-means partition.',
				dark_mode=dark_tables, precision=0, )
			
			if Z.shape[ 1 ] >= 2:
				markers = cfg.MARKERS
				fig4, ax4 = plt.subplots( figsize=(8, 6) )
				for lbl in np.unique( labels ):
					idx = labels == lbl
					color = cfg.PALETTE[ int( lbl ) % len( cfg.PALETTE ) ]
					marker = markers[ int( lbl ) % len( markers ) ]
					ax4.scatter( Z[ idx, 0 ], Z[ idx, 1 ], s=30, alpha=0.9, edgecolors='#020617',
						linewidths=0.6, marker=marker, label=f'Cluster {lbl}', c=[ color ], )
				ax4.set_xlabel( 'PC1' )
				ax4.set_ylabel( 'PC2' )
				ax4.set_title( 'k-Means Clusters in PCA Space' )
				ax4.grid( True, alpha=0.25 )
				ax4.legend( title="Cluster", fontsize=8 )
				st.pyplot( fig4 )
				st.caption( 'Distinct shapes and colors make clusters easier to separate visually.' )

# =============================================================================
# Feature Engineering
# =============================================================================
with tabs[ 4 ]:
	if not numeric:
		st.warning( 'Feature engineering operates on numeric columns.' )
	else:
		default_fe = numeric_default[ : min( 10, len( numeric_default ) ) ]
		fe_cols = st.multiselect(
			'Numeric features to transform',
			numeric,
			default=default_fe,
			key='fe_cols',
		)
		
		if not fe_cols:
			st.info( 'Select at least one numeric feature.' )
		else:
			c1, c2, c3 = st.columns( 3 )
			with c1:
				impute_strats = st.multiselect(
					'Imputation strategy (in order)',
					[ 'mean', 'median', 'most_frequent', 'constant', 'knn' ],
					default=[ 'median' ],
					key='fe_impute',
				)
				const_fill = st.number_input(
					'Constant fill value', value=0.0, key='fe_const'
				)
			with c2:
				do_winsor = st.checkbox(
					'Winsorize (cap extremes)', value=False, key='fe_winsor'
				)
				w_lim = st.slider(
					'Winsor tail limit', 0.0, 0.25, 0.01, 0.005, key='fe_wlim'
				)
			with c3:
				scaler_name = st.selectbox(
					'Scaling',
					[ 'None', 'Standard', 'MinMax', 'Robust' ],
					index=1,
					key='fe_scaler',
				)
				log1p = st.checkbox(
					'Log1p transform (positive-only)', value=False, key='fe_log1p'
				)
			
			X = df[ fe_cols ].apply( pd.to_numeric, errors="coerce" )
			
			X_imp = X.copy( )
			for s in impute_strats:
				if s == 'knn':
					knn = KNNImputer( n_neighbors=5, weights='distance' )
					X_imp = pd.DataFrame( knn.fit_transform( X_imp ), columns=fe_cols )
				elif s == 'constant':
					simp = SimpleImputer(
						strategy='constant', fill_value=float( const_fill )
					)
					X_imp = pd.DataFrame( simp.fit_transform( X_imp ), columns=fe_cols )
				else:
					simp = SimpleImputer( strategy=s )
					X_imp = pd.DataFrame( simp.fit_transform( X_imp ), columns=fe_cols )
			
			X_win = X_imp.copy( )
			if do_winsor and w_lim > 0.0:
				for c in fe_cols:
					v = X_win[ c ].values.astype( float )
					X_win[ c ] = winsorize(
						v, limits=(float( w_lim ), float( w_lim ))
					)
			
			X_log = X_win.copy( )
			if log1p:
				for c in fe_cols:
					v = X_log[ c ].values.astype( float )
					if np.nanmin( v ) <= -1.0:
						continue
					X_log[ c ] = np.log1p( v )
			
			if scaler_name == 'Standard':
				sc = StandardScaler( )
				X_out = pd.DataFrame( sc.fit_transform( X_log.values ), columns=fe_cols )
			elif scaler_name == 'MinMax':
				sc = MinMaxScaler( )
				X_out = pd.DataFrame( sc.fit_transform( X_log.values ), columns=fe_cols )
			elif scaler_name == 'Robust':
				sc = RobustScaler( )
				X_out = pd.DataFrame( sc.fit_transform( X_log.values ), columns=fe_cols )
			else:
				X_out = X_log.copy( )
			
			st.subheader( 'Transformed Data Preview' )
			render_table(
				X_out.head( preview_rows ),
				caption='Feature matrix after imputation, optional winsorization/log transform, '
				        'and scaling.',
				dark_mode=dark_tables,
				precision=4,
			)
			
			st.subheader( 'Before vs After (Summary Metrics)' )
			before = descriptive_profile( df, fe_cols )[
				[ 'feature', 'mean', 'std', 'skew', 'kurtosis', 'outlier_iqr_pct' ]
			]
			after_df = descriptive_profile( X_out, fe_cols )[
				[ 'feature', 'mean', 'std', 'skew', 'kurtosis', 'outlier_iqr_pct' ]
			]
			merged = before.merge(
				after_df, on='feature', suffixes=('_before', '_after')
			)
			render_table(
				merged,
				caption='Compare distribution metrics before/after transformations.',
				dark_mode=dark_tables,
				precision=4,
			)
			
			st.session_state[ 'X_fe' ] = X_out
			st.session_state[ 'X_fe_cols' ] = fe_cols

# =============================================================================
# Anomaly Detection
# =============================================================================
with tabs[ 5 ]:
	if not numeric:
		st.warning( 'Requires numeric columns for anomaly detection.' )
	else:
		default_ad = numeric_default[ : min( 10, len( numeric_default ) ) ]
		ad_cols = st.multiselect(
			'Numeric columns used for detection',
			numeric,
			default=default_ad,
			key='ad_cols',
		)
		methods = st.multiselect(
			'Detection methods',
			[
					'Z-score (univariate)',
					'IQR rule (univariate)',
					'MAD robust z (univariate)',
					'Isolation Forest (multivariate)',
			],
			default=[
					'Z-score (univariate)',
					'IQR rule (univariate)',
					'Isolation Forest (multivariate)',
			],
			key='ad_methods',
		)
		
		if not ad_cols:
			st.info( 'Select at least one numeric column.' )
		else:
			X = df[ ad_cols ].apply( pd.to_numeric, errors='coerce' )
			X = X.fillna( X.median( numeric_only=True ) )
			
			flags = pd.DataFrame( index=df.index )
			summary: List[ Dict[ str, Any ] ] = [ ]
			
			if 'Z-score (univariate)' in methods:
				z_thr = st.slider(
					'Z-score threshold', 2.0, 6.0, 3.0, 0.1, key='ad_z_thr'
				)
				zf = np.zeros( len( df ), dtype=bool )
				for c in ad_cols:
					v = X[ c ].values.astype( float )
					z = (v - v.mean( )) / (v.std( ddof=0 ) + 1e-12)
					zf |= np.abs( z ) > float( z_thr )
				flags[ 'z_flag' ] = zf
				summary.append(
					{
							'method': 'Z-score',
							'flagged': int( zf.sum( ) ),
							'flagged_pct': float( zf.mean( ) * 100.0 ),
					}
				)
			
			if 'IQR rule (univariate)' in methods:
				k = st.slider( 'IQR multiplier', 1.0, 5.0, 1.5, 0.1, key='ad_iqr_k' )
				iqf = np.zeros( len( df ), dtype=bool )
				for c in ad_cols:
					v = X[ c ].values.astype( float )
					q1 = np.percentile( v, 25 )
					q3 = np.percentile( v, 75 )
					iqr = q3 - q1
					lo = q1 - float( k ) * iqr
					hi = q3 + float( k ) * iqr
					iqf |= (v < lo) | (v > hi)
				flags[ 'iqr_flag' ] = iqf
				summary.append(
					{
							'method': 'IQR',
							'flagged': int( iqf.sum( ) ),
							'flagged_pct': float( iqf.mean( ) * 100.0 ),
					}
				)
			
			if 'MAD robust z (univariate)' in methods:
				rz_thr = st.slider( 'Robust z threshold', 2.0, 8.0, 3.5, 0.1, key='ad_rz_thr' )
				mf = np.zeros( len( df ), dtype=bool )
				for c in ad_cols:
					v = X[ c ].values.astype( float )
					med = np.median( v )
					mad = np.median( np.abs( v - med ) ) + 1e-12
					rz = 0.6745 * (v - med) / mad
					mf |= np.abs( rz ) > float( rz_thr )
				flags[ 'mad_flag' ] = mf
				summary.append(
					{
						'method': 'MAD robust z',
						'flagged': int( mf.sum( ) ),
						'flagged_pct': float( mf.mean( ) * 100.0 ),
					} )
			
			if 'Isolation Forest (multivariate)' in methods:
				cont = st.slider( 'IsolationForest contamination', 0.001, 0.20, 0.02, 0.001,
					key='ad_iso_cont', )
				iso = IsolationForest( random_state=42, contamination=float( cont ) )
				pred = iso.fit_predict( X.values )
				isof = pred == -1
				flags[ 'iso_flag' ] = isof
				summary.append(
					{
						'method': 'IsolationForest',
						'flagged': int( isof.sum( ) ),
						'flagged_pct': float( isof.mean( ) * 100.0 ),
					} )
			
			st.subheader( 'Detection Summary' )
			render_table( pd.DataFrame( summary ),
				caption='Univariate rules flag extremes per feature; multivariate methods flag '
				        'unusual combinations.', dark_mode=dark_tables, precision=4, )
			
			combined = flags.any( axis=1 ) if not flags.empty else pd.Series(
				False, index=df.index )
			flagged_rows = df.loc[ combined, ad_cols ]
			st.subheader( 'Flagged Rows (Any Method)' )
			render_table( flagged_rows.head( 300 ),
				caption='Validate whether flagged rows represent data issues or true anomalies.',
				dark_mode=dark_tables, precision=4, max_rows=300, )

# =============================================================================
# Modeling
# =============================================================================
with (tabs[ 6 ]):
	if not numeric and not categorical:
		st.warning( 'Modeling requires at least one numeric or categorical column.' )
	else:
		task = st.radio( 'Task type', [ 'Regression', 'Classification' ], index=0,
			horizontal=True, key='mdl_task', )
		test_size = st.slider( 'Test size', 0.10, 0.50, 0.20, 0.05, key='mdl_test_size' )
		rs = st.number_input( 'Random state', value=42, step=1, key='mdl_rs' )
		
		if task == 'Regression':
			if not numeric:
				st.warning( 'Regression requires numeric target and features.' )
			else:
				target = st.selectbox( 'Numeric target', numeric,
					index=( numeric.index( numeric_default[ 0 ] )
							if numeric_default and numeric_default[ 0 ] in numeric
							else 0 ), key='mdl_reg_target', )
				feat_opts = [ c for c in numeric if c != target ]
				default_reg_feats = [
						c for c in numeric_default if c in feat_opts
				][ : min( 10, len( feat_opts ) ) ]
				if not default_reg_feats:
					default_reg_feats = feat_opts[ : min( 10, len( feat_opts ) ) ]
				
				features = st.multiselect( 'Numeric features', feat_opts, default=default_reg_feats,
					key='mdl_reg_features', )
				
				if not features:
					st.info( 'Select at least one numeric feature.' )
				else:
					X = df[ features ].apply( pd.to_numeric, errors='coerce' )
					X = X.fillna( X.median( numeric_only=True ) )
					y = pd.to_numeric( df[ target ], errors='coerce' ).values
					X_train, X_test, y_train, y_test = train_test_split( X.values, y,
						test_size=float( test_size ), random_state=int( rs ), shuffle=True, )
					
					models = \
					{
						'LinearRegression': LinearRegression( ),
						'Ridge': Ridge( ),
						'Lasso': Lasso( ),
						'ElasticNet': ElasticNet( ),
						'RandomForestRegressor': RandomForestRegressor( random_state=42 ),
					}
					
					sel_models = st.multiselect( 'Models to train', list( models.keys( ) ),
						default=[ 'LinearRegression', 'Ridge', 'RandomForestRegressor', ],
						key='mdl_reg_models', )
					
					if st.button( 'Train regression models', type='primary',
							key='mdl_reg_train', ):
						results: List[ Dict[ str, Any ] ] = [ ]
						fitted: Dict[ str, Any ] = { }
						
						for name in sel_models:
							m = models[ name ]
							m.fit( X_train, y_train )
							preds = m.predict( X_test )
							
							results.append(
								{
										'model': name,
										'rmse': float( mean_squared_error( y_test, preds,
											squared=False ) ),
										'mae': float( mean_absolute_error( y_test, preds ) ),
										'r2': float( r2_score( y_test, preds ) ),
								}
							)
							fitted[ name ] = (m, preds, y_test)
						
						res_df = pd.DataFrame( results ).sort_values( 'rmse', ascending=True )
						render_table( res_df, title='Regression model comparison',
							caption='Lower RMSE/MAE and higher R² indicate better performance.',
							dark_mode=dark_tables, precision=6, )
						
						best_name = str( res_df.iloc[ 0 ][ "model" ] )
						best_model, best_preds, best_y = fitted[ best_name ]
						
						fig, ax = plt.subplots( figsize=(6, 5) )
						styled_scatter( ax, best_y, best_preds, series_index=2, size=32 )
						ax.plot( [ best_y.min( ), best_y.max( ) ], [ best_y.min( ), best_y.max(
						
						) ],
							'r--', linewidth=1.2, )
						ax.set_xlabel( 'Actual' )
						ax.set_ylabel( 'Predicted' )
						ax.set_title( f'Actual vs Predicted — {best_name}' )
						st.pyplot( fig )
						st.caption( 'Points near the diagonal indicate accurate predictions ' )
						
						st.session_state[ 'last_model_payload' ] = {
								'task': 'Regression',
								'name': best_name,
								'model': best_model,
								'X_test': X_test,
								'y_test': best_y,
								'preds': best_preds,
						}
		
		else:  # Classification
			if not categorical or not numeric:
				st.warning( 'Classification requires a categorical target and numeric features.' )
			else:
				default_target_idx = 0
				if categorical_default:
					first_default = categorical_default[ 0 ]
					if first_default in categorical:
						default_target_idx = categorical.index( first_default )
				
				target = st.selectbox(
					'Categorical target',
					categorical,
					index=default_target_idx,
					key='mdl_clf_target',
				)
				
				default_clf_feats = numeric_default[ : min( 10, len( numeric_default ) ) ]
				features = st.multiselect(
					'Numeric features',
					numeric,
					default=default_clf_feats,
					key='mdl_clf_features',
				)
				
				if not features:
					st.info( 'Select at least one numeric feature.' )
				else:
					X = df[ features ].apply( pd.to_numeric, errors='coerce' )
					X = X.fillna( X.median( numeric_only=True ) )
					y_raw = df[ target ].astype( str ).fillna( '(missing)' )
					le = LabelEncoder( )
					y = le.fit_transform( y_raw.values )
					
					strat = y if len( np.unique( y ) ) > 1 else None
					X_train, X_test, y_train, y_test = train_test_split(
						X.values,
						y,
						test_size=float( test_size ),
						random_state=int( rs ),
						shuffle=True,
						stratify=strat,
					)
					
					models = {
							'LogisticRegression': LogisticRegression( max_iter=4000,
								random_state=42 ),
							'RandomForestClassifier': RandomForestClassifier( random_state=42 ), }
					sel_models = st.multiselect( 'Models to train', list( models.keys( ) ),
						default=[ 'LogisticRegression', 'RandomForestClassifier', ],
						key="mdl_clf_models", )
					
					if st.button( 'Train classification models', type='primary',
							key='mdl_clf_train', ):
						results: List[ Dict[ str, Any ] ] = [ ]
						fitted: Dict[ str, Any ] = { }
						
						for name in sel_models:
							m = models[ name ]
							m.fit( X_train, y_train )
							preds = m.predict( X_test )
							acc = float( accuracy_score( y_test, preds ) )
							auc = np.nan
							if hasattr( m, 'predict_proba' ) and len( np.unique( y_test ) ) == 2:
								p1 = m.predict_proba( X_test )[ :, 1 ]
								try:
									auc = float( roc_auc_score( y_test, p1 ) )
								except Exception:
									auc = np.nan
							
							results.append( { 'model': name, 'accuracy': acc, 'auc': auc } )
							fitted[ name ] = (m, preds, y_test)
						
						res_df = pd.DataFrame( results ).sort_values( 'accuracy', ascending=False )
						render_table( res_df, title='Classification model comparison',
							caption='Higher accuracy and AUC (binary) indicate better '
							        'performance.', dark_mode=dark_tables, precision=6, )
						
						best_name = str( res_df.iloc[ 0 ][ "model" ] )
						best_model, best_preds, best_y = fitted[ best_name ]
						cm = confusion_matrix( best_y, best_preds )
						fig, ax = plt.subplots( figsize=(6, 4) )
						sns.heatmap( cm, annot=True, fmt="d", cmap='Blues',
							cbar=False, linewidths=0.4, linecolor='#0f172a',
							annot_kws={ 'size': 8 }, ax=ax, )
						ax.set_title( f'Confusion Matrix — {best_name}' )
						ax.set_xlabel( 'Predicted' )
						ax.set_ylabel( 'Actual' )
						st.pyplot( fig )
						st.caption(
							'Diagonal cells are correct classifications; off-diagonals are '
							'misclassifications.'
						)
						
						st.session_state[ 'last_model_payload' ] = {
								'task': 'Classification',
								'name': best_name,
								'model': best_model,
								'X_test': X_test,
								'y_test': best_y,
								'preds': best_preds,
								'labels': le.classes_.tolist( ),
						}

# =============================================================================
#  Diagnostics
# =============================================================================
with tabs[ 7 ]:
	payload = st.session_state.get( 'last_model_payload' )
	if not payload:
		st.info( 'Train a model in the Modeling tab to see diagnostics.' )
	else:
		task = payload[ 'task' ]
		name = payload[ 'name' ]
		model = payload[ 'model' ]
		X_test = payload[ 'X_test' ]
		y_test = payload[ 'y_test' ]
		preds = payload[ 'preds' ]
		
		st.markdown( f'##### Best Model: {name} ({task})' )
		
		if task == 'Regression':
			resid = y_test - preds
			metrics_df = pd.DataFrame(
				[ {
						'rmse': float( mean_squared_error( y_test, preds, squared=False ) ),
						'mae': float( mean_absolute_error( y_test, preds ) ),
						'r2': float( r2_score( y_test, preds ) ),
						'resid_mean': float( np.mean( resid ) ),
						'resid_std': float( np.std( resid ) ),
				  } ]
			)
			render_table( metrics_df,
				caption=(
						'Residual diagnostics: residual mean near 0 and stable spread '
						'suggest less bias and more homoscedastic errors.'),
				dark_mode=dark_tables,
				precision=6,
			)
			
			fig, ax = plt.subplots( figsize=(8, 4) )
			ax.hist( resid, bins=40, edgecolor='#020617', linewidth=0.6,
				color=cfg.PALETTE[ 3 ], alpha=0.9,
			)
			ax.set_title( 'Residual Distribution' )
			ax.set_xlabel( 'Residual' )
			ax.set_ylabel( 'Count' )
			ax.grid( True, alpha=0.25 )
			st.pyplot( fig )
			st.caption(
				'A symmetric, centered residual distribution indicates fewer systematic errors.'
			)
			
			fig2, ax2 = plt.subplots( figsize=(8, 4) )
			styled_scatter( ax2, preds, resid, series_index=4, size=30 )
			ax2.axhline( 0, linestyle='--', color='#f97316', linewidth=1.2 )
			ax2.set_title( 'Residuals vs Predicted' )
			ax2.set_xlabel( 'Predicted' )
			ax2.set_ylabel( 'Residual' )
			st.pyplot( fig2 )
			st.caption(
				'Patterns (e.g., funnel shapes) indicate heteroscedasticity; non-random structure '
				'suggests mis-specification.'
			)
		
		else:  # Classification
			labels = payload.get( "labels" )
			rep = classification_report(
				y_test, preds, output_dict=True, zero_division=0
			)
			rep_df = pd.DataFrame( rep ).T
			render_table(
				rep_df,
				title='Classification Report',
				caption=(
						'Precision, recall, and F1 scores by class highlight which labels '
						'the model struggles with.'
				),
				dark_mode=dark_tables,
				precision=4,
			)
			
			cm = confusion_matrix( y_test, preds )
			fig, ax = plt.subplots( figsize=(6, 4) )
			sns.heatmap( cm, annot=True, fmt='d', cmap='Blues', cbar=False,
				linewidths=0.4, linecolor='#0f172a', annot_kws={ 'size': 8 }, ax=ax, )
			ax.set_title( 'Confusion Matrix' )
			ax.set_xlabel( 'Predicted' )
			ax.set_ylabel( 'Actual' )
			st.pyplot( fig )
			st.caption(
				'Diagonal elements are correct classifications; off-diagonal entries indicate '
				'confusion between classes.'
			)
			
			if hasattr( model, 'predict_proba' ) and len( np.unique( y_test ) ) == 2:
				p1 = model.predict_proba( X_test )[ :, 1 ]
				fpr, tpr, _ = roc_curve( y_test, p1 )
				auc = float( roc_auc_score( y_test, p1 ) )
				fig2, ax2 = plt.subplots( figsize=(6, 4) )
				ax2.plot( fpr, tpr, label=f'ROC (AUC={auc:.3f})',
					linewidth=1.4, color=cfg.PALETTE[ 5 ], )
				ax2.plot( [ 0, 1 ], [ 0, 1 ], 'r--',
					linewidth=1.0, label='Random', )
				ax2.set_xlabel( 'False Positive Rate' )
				ax2.set_ylabel( 'True Positive Rate' )
				ax2.set_title( 'ROC Curve' )
				ax2.grid( True, alpha=0.25 )
				ax2.legend( )
				st.pyplot( fig2 )
				st.caption(
					'Higher curves and AUC closer to 1 indicate stronger ranking performance.'
				)
				
				prec, rec, _ = precision_recall_curve( y_test, p1 )
				fig3, ax3 = plt.subplots( figsize=(6, 4) )
				ax3.plot( rec, prec, linewidth=1.4, color=cfg.PALETTE[ 6 ], )
				ax3.set_xlabel( 'Recall' )
				ax3.set_ylabel( 'Precision' )
				ax3.set_title( 'Precision–Recall Curve' )
				ax3.grid( True, alpha=0.25 )
				st.pyplot( fig3 )
				st.caption(
					'Precision–Recall curves are especially informative for imbalanced classes.' )

st.markdown('---')
st.caption( ' Sake' )
