######  sake
![](https://github.com/is-leeroy-jenkins/Sake/blob/master/resources/assets/sake_project.png)
- A modular machine learning framework for budget execution & data analysis built in Python with **Scikit**, **XGBoost**, **PyTorch**, and **TensorFlow**. Designed for rapid experimentation, visualization, and benchmarking of both **classification** and **regression** models, it provides a structured yet extensible workflow that’s equally useful for teaching, prototyping, and real-world application development.

## Demo
![](https://github.com/is-leeroy-jenkins/Sake/blob/master/resources/assets/sake-demo.gif)

## ☁️ Google (Cloud)
<a href="https://colab.research.google.com/github/is-leeroy-jenkins/sake/blob/master/models.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

![](https://github.com/is-leeroy-jenkins/Sake/blob/master/resources/assets/Sake-nb.gif)

## 🧱 Databricks
[![Databricks Notebook](https://img.shields.io/badge/Databricks%20Repo-Sake--Py-FF3621?logo=databricks&logoColor=white)](https://dbc-a0c21f80-7bb3.cloud.databricks.com/editor/notebooks/1460524320197777?o=7474645703081351)
- A data engineering, analytics, and artificial intelligence collaborative workspace
- Codebase


  
## 🕸️ Streamlit (Web)

[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://sake-py.streamlit.app/)
![](https://github.com/is-leeroy-jenkins/Sake/blob/master/resources/assets/sake-demo.gif)

## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```


### Option A — Google Colab (no local setup)

```
1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.
```

### Option B — Local (conda or venv)

```
bash
# 1) Create environment
conda create -n sake python=3.11 -y
conda activate sake

# 2) Install dependencies
pip install -U pip wheel setuptools
pip install pandas numpy scipy matplotlib seaborn scikit-learn jupyter

# 3) Launch Jupyter
jupyter notebook
```

Open `ipynb/schedule-x.ipynb` and run cells top-to-bottom.


- This section describes how to clone the repository, install dependencies, and launch the **Sake**
Streamlit application locally. 
- The Streamlit app provides an interactive interface for exploring
Account Balances data, performing statistical analysis, and training machine-learning models as
described in this project.

### 📥 Clone the Repository

First, clone the Sake repository from GitHub and navigate into the project directory:

```bash
git clone https://github.com/<your-username>/sake.git
cd sake
```



### 🐍 Create a Virtual Environment (Recommended)

It is strongly recommended to run Sake in an isolated Python virtual environment.

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Verify that the virtual environment is active before proceeding.



### 📦 Install Dependencies

Install all required Python packages using the provided `requirements.txt` file:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note**
> Ensure that `streamlit` is included in `requirements.txt`. If not, install it manually:
>
> ```bash
> pip install streamlit
> ```



### ▶️ Launch the Streamlit Application

Once dependencies are installed, start the Streamlit app by running:

```bash
streamlit run app.py
```

Streamlit will start a local development server and automatically open the application in your
default web browser. If it does not open automatically, the terminal will display a local URL
similar to:

```
http://localhost:8501
```

Open that address in your browser to access the app.


## 🔬 Data Source
- File A (Account Balances) published monthly by agencies on [USASpending](https://www.usaspending.gov/download_center/custom_account_data?about-the-data=file-a)
- Required by the DATA Act.
- Pulled automatically from data in the [Governmentwide Treasury Account Symbol Adjusted Trial Balance System (GTAS)](https://www.fiscal.treasury.gov/gtas/)
- Contains Budgetary resources, obligation, and outlay data for all the relevant [Treasury Account Symbols (TAS)](https://tfx.treasury.gov/taxonomy/term/10257) in a reporting agency.
- It includes both award and non-award spending (grouped together), and crosswalks with the SF 133 report.

## 🔄 Unified Evaluation Pipeline
#### A single interface `train_and_evaluate()` to:
- Train models
- Cross-validate with nested k-fold
- Generate predictions
- Output evaluation plots & performance metrics
- Store results & timings for meta-analysis


### 📊 Using the Application

After launch, the Streamlit interface will guide you through:

1. **Uploading File A (Account Balances)**
   Upload an Excel file containing GTAS / USASpending Account Balances data.

2. **Exploring the Data**
   View previews, column summaries, descriptive statistics, and distributions.

3. **Statistical Analysis**
   Perform correlation analysis, hypothesis testing, and ANOVA-style comparisons.

4. **Feature Engineering**
   Apply dimensionality reduction techniques such as PCA, Truncated SVD, or Factor Analysis.

5. **Model Training & Evaluation**
   Train and evaluate regression, classification, or clustering models using a unified
   `train_and_evaluate()` pipeline with optional cross-validation and diagnostics.



### 🛑 Stopping the App

To stop the Streamlit server, return to the terminal and press:

```
Ctrl + C
```

---

### 🧪 Supported Python Versions

* Python **3.9 – 3.12** (64-bit recommended)
* Tested on Windows and macOS



If you want, I can also:

* Add a **Docker-based run section**
* Add **screenshots / UI walkthrough**
* Align this section stylistically with another README you’ve already approved

## 📊 Rich Visualization Toolkit
- Confusion Matrix Heatmaps 🔥
- ROC & Precision-Recall Curves 📈
- Actual vs. Predicted Scatterplots 🎯
- Residual Analysis & Error Distribution 🎭
- Feature Importance Charts 📊

## ⏱️ Timing & Benchmarking
- Automatically logs `fit` and `predict` durations
- Model performance rankings across tasks
- Output available in tabular format for export

## 💡 Custom Dataset Support
- Accepts CSVs, Excel files, or Pandas DataFrames
- Label encoding, numeric coercion, missing data handling
- Drop-in replacement for datasets via parameter injection

## 🧪 Research Ready
- Benchmark dozens of models easily
- Plug-in architecture for testing experimental models
- Use in classrooms to demo interpretability, overfitting, and variance



## 📊 Descriptive Statistics

| Statistic         | Description                             | Use in Budget Analysis                                               |
|------------------|-----------------------------------------|----------------------------------------------------------------------|
| **Mean**         | Average value                           | Avg. Outlays, Obligations, etc., across accounts                |
| **Median**       | Middle value                            | Robust central tendency in skewed financial data                    |
| **Mode**         | Most frequent value                     | Identify common MainAccountCodes or Availability categories     |
| **Standard Deviation** | Spread around the mean                | Indicates variability in execution rates or balances                |
| **Variance**     | Square of standard deviation            | Used in statistical tests and model diagnostics                     |
| **Range**        | Difference between max and min          | Measures total spread of financial metrics                          |
| **Interquartile Range (IQR)** | Spread of middle 50% of data           | Identifies budget outliers and extreme accounts                     |
| **Skewness**     | Asymmetry of distribution               | Skewed obligations suggest few accounts dominate totals             |
| **Kurtosis**     | "Peakedness" of distribution            | High values indicate outlier-prone financial data                   |





## 🔍 Inferrential Statistics


| Metric           | Description                                            | Use in Budget Analysis                                               |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------------|
| **Pearson Correlation** | Linear relationship between variables                  | E.g., TotalResources vs. Obligations                                 |
| **Spearman Correlation**| Monotonic (rank-based) relationship                    | More robust to non-linear trends in financial execution              |
| **t-test**              | Compare means between 2 groups                         | Discretionary vs. Mandatory accounts' execution rates                |
| **ANOVA**               | Compare means across multiple groups                   | Obligations across availability periods or account types             |
| **Chi-square Test**     | Categorical independence                               | Are Main Account Codes related to availability or a specific agency? |
| **Confidence Intervals**| Estimate range of a population mean                    | Upper and lower bound expected obligations or recoveries             |
| **Regression Coefficients (p-values)** | Test variable significance                             | Are Recoveries a significant predictor of UnobligatedBalance?        |
| **F-statistic (overall regression)**   | Test whole model fit                                   | Determines the combined influence of all predictors                  |
| **Z-score / Outlier Tests** | Deviation from standard mean                           | Identify abnormal balances or lapse rates                            |
| **Boxplots**            | Visual outlier detection                               | Discover obligation anomalies within agencies                        |


## ✅ Classification:
| Model                    | Module                               |
|-------------------------|--------------------------------------|
| Logistic Regression     | `sklearn.linear_model.LogisticRegression` |
| SVM                     | `sklearn.svm.SVC`                    |
| Decision Tree           | `sklearn.tree.DecisionTreeClassifier`|
| Random Forest           | `sklearn.ensemble.RandomForestClassifier`|
| XGBoost Classifier      | `xgboost.XGBClassifier`              |
| K-Nearest Neighbors     | `sklearn.neighbors.KNeighborsClassifier`|
| Gaussian Naive Bayes    | `sklearn.naive_bayes.GaussianNB`     |
| **Extra Trees**         | `sklearn.ensemble.ExtraTreesClassifier`|
| **Bagging**             | `sklearn.ensemble.BaggingClassifier`|
| **AdaBoost**            | `sklearn.ensemble.AdaBoostClassifier`|


## 📉 Regression:
| Model                        | Module                                   |
|-----------------------------|------------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression`  |
| Ridge Regression            | `sklearn.linear_model.Ridge`             |
| Lasso Regression            | `sklearn.linear_model.Lasso`             |
| ElasticNet                  | `sklearn.linear_model.ElasticNet`        |
| Support Vector Regressor    | `sklearn.svm.SVR`                        |
| Decision Tree Regressor     | `sklearn.tree.DecisionTreeRegressor`     |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor`|
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor`|
| XGBoost Regressor           | `xgboost.XGBRegressor`                   |
| K-Nearest Neighbors         | `sklearn.neighbors.KNeighborsRegressor` |
| **AdaBoost Regressor**      | `sklearn.ensemble.AdaBoostRegressor`    |
| **Extra Trees Regressor**   | `sklearn.ensemble.ExtraTreesRegressor`  |



## 📦 Dependencies

| Package          | Description                                                      | Link                                                  |
|------------------|------------------------------------------------------------------|-------------------------------------------------------|
| numpy            | Numerical computing library                                      | [numpy.org](https://numpy.org/)                      |
| pandas           | Data manipulation and DataFrames                                 | [pandas.pydata.org](https://pandas.pydata.org/)      |
| matplotlib       | Plotting and visualization                                       | [matplotlib.org](https://matplotlib.org/)            |
| seaborn          | Statistical data visualization                                   | [seaborn.pydata.org](https://seaborn.pydata.org/)    |
| scikit-learn     | ML modeling and metrics                                          | [scikit-learn.org](https://scikit-learn.org/stable/) |
| xgboost          | Gradient boosting framework (optional)                          | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| torch            | PyTorch deep learning library                                    | [pytorch.org](https://pytorch.org/)                  |
| tensorflow       | End-to-end ML platform                                           | [tensorflow.org](https://www.tensorflow.org/)        |
| openai           | OpenAI’s Python API client                                       | [openai-python](https://github.com/openai/openai-python) |
| requests         | HTTP requests for API and web access                             | [requests.readthedocs.io](https://requests.readthedocs.io/) |
| PySimpleGUI      | GUI framework for desktop apps                                   | [pysimplegui.readthedocs.io](https://pysimplegui.readthedocs.io/) |
| typing           | Type hinting standard library                                    | [typing Docs](https://docs.python.org/3/library/typing.html) |
| pyodbc           | ODBC database connector                                          | [pyodbc GitHub](https://github.com/mkleehammer/pyodbc) |
| fitz             | PDF document parser via PyMuPDF                                  | [pymupdf](https://pymupdf.readthedocs.io/)           |
| pillow           | Image processing library                                         | [python-pillow.org](https://python-pillow.org/)       |
| openpyxl         | Excel file processing                                            | [openpyxl Docs](https://openpyxl.readthedocs.io/)     |
| soundfile        | Read/write sound file formats                                    | [pysoundfile](https://pysoundfile.readthedocs.io/)    |
| sounddevice      | Audio I/O interface                                              | [sounddevice Docs](https://python-sounddevice.readthedocs.io/) |
| loguru           | Structured, elegant logging                                      | [loguru GitHub](https://github.com/Delgan/loguru)     |
| statsmodels      | Statistical tests and regression diagnostics                     | [statsmodels.org](https://www.statsmodels.org/)       |
| dotenv           | Load environment variables from `.env`                          | [python-dotenv GitHub](https://github.com/theskumar/python-dotenv) |
| python-dotenv    | Same as above (modern usage)                                     | [python-dotenv](https://saurabh-kumar.com/python-dotenv/) |




## 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```


## 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis



> **Disclaimer**: This is for analytical exploration, research, and education purposes.  
> This is **not** an official government product; validate against authoritative sources before use.

## 📝 License

Sake is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Sake/blob/master/LICENSE.txt).


---

