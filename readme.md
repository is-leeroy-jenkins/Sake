####  Sake
![](https://github.com/is-leeroy-jenkins/sake/blob/master/resources/assets/img/git/SakeProject.png)
- A modular machine learning framework for budget execution & data analysis built in Python with **Scikit**, **XGBoost**, **PyTorch**, and **TensorFlow**. Designed for rapid experimentation, visualization, and benchmarking of both **classification** and **regression** models, it provides a structured yet extensible workflow that’s equally useful for teaching, prototyping, and real-world application development.
<a href="https://colab.research.google.com/github/is-leeroy-jenkins/sake/blob/master/models.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 🔬 Data Source
- File A (Account Balances) published monthly by agencies on [USASpending](https://www.usaspending.gov/download_center/custom_account_data?about-the-data=file-a)
- Required by the DATA Act.
- Pulled automatically from data in the [Governmentwide Treasury Account Symbol Adjusted Trial Balance System (GTAS)](https://www.fiscal.treasury.gov/gtas/)
- Contains Budgetary resources, obligation, and outlay data for all the relevant [Treasury Account Symbols (TAS)](https://tfx.treasury.gov/taxonomy/term/10257) in a reporting agency.
- It includes both award and non-award spending (grouped together), and crosswalks with the SF 133 report.
  
## 🚀 Features

### 🔄 Unified Evaluation Pipeline
A single interface `train_and_evaluate()` to:
- Train models
- Cross-validate with nested k-fold
- Generate predictions
- Output evaluation plots & performance metrics
- Store results & timings for meta-analysis

## 🎯 Quickstart

### Option A — Google Colab (no local setup)

1. Click the **Open In Colab** badge above.
2. Upload your CSV or mount Google Drive.
3. Set `DATA_PATH` near the top of the notebook.
4. **Runtime → Run all**.

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

### 📊 Rich Visualization Toolkit
- Confusion Matrix Heatmaps 🔥
- ROC & Precision-Recall Curves 📈
- Actual vs. Predicted Scatterplots 🎯
- Residual Analysis & Error Distribution 🎭
- Feature Importance Charts 📊

### ⏱️ Timing & Benchmarking
- Automatically logs `fit` and `predict` durations
- Model performance rankings across tasks
- Output available in tabular format for export

### 💡 Custom Dataset Support
- Accepts CSVs, Excel files, or Pandas DataFrames
- Label encoding, numeric coercion, missing data handling
- Drop-in replacement for datasets via parameter injection

### 🧪 Research & Education Friendly
- Benchmark dozens of models easily
- Plug-in architecture for testing experimental models
- Use in classrooms to demo interpretability, overfitting, and variance



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



## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```



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



## 🔮 Roadmap

- [ ] Add time series models (Prophet, ARIMA)
- [ ] Integrate GridSearchCV for model tuning
- [ ] SHAP-based interpretability
- [ ] Flask/FastAPI API for deploying forecasts
- [ ] LLM summarization of forecast outcomes



## 🤝 Contributing

1. 🍴 Fork the project
2. 🔧 Create a branch: `git checkout -b feat/new-feature`
3. ✅ Commit and push changes
4. 📬 Submit a pull request

> **Disclaimer**: This is for analytical exploration, research, and education purposes.  
> This is **not** an official government product; validate against authoritative sources before use.

## 📝 License

Sake is published under the [MIT General Public License v3](https://github.com/is-leeroy-jenkins/Sake/blob/master/LICENSE.txt).


---

