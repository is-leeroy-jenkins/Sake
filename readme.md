####  Sake
![](https://github.com/is-leeroy-jenkins/Sake/blob/master/resources/assets/img/git/SakeProject.png)
- Sake is your go-to, modular machine learning framework for Budget Execution data analysis built in Python with **Scikit**, **XGBoost**, **PyTorch**, and **TensorFlow**. Designed for rapid experimentation, visualization, and benchmarking of both **classification** and **regression** models, it provides a structured yet extensible workflow that’s equally useful for teaching, prototyping, and real-world application development.
<a href="https://colab.research.google.com/github/is-leeroy-jenkins/Sake/blob/main/ipynb/outlays.ipynb" target="_parent">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 🔬 Data Source
- File A (Account Balances) published monthly by agencies on ![USASpending](https://www.usaspending.gov/download_center/custom_account_data?about-the-data=file-a)
- Required by the DATA Act.
- Pulled automatically from data in the ![Governmentwide Treasury Account Symbol Adjusted Trial Balance System (GTAS)](https://www.fiscal.treasury.gov/gtas/)
- Contains Budgetary resources, obligation, and outlay data for all the relevant ![Treasury Account Symbols (TAS)](https://tfx.treasury.gov/taxonomy/term/10257) in a reporting agency.
- It includes both award and non-award spending (grouped together), and crosswalks with the SF 133 report.
  
## 🚀 Features

### 🔄 Unified Evaluation Pipeline
Easily run multiple models through a single function `train_and_evaluate()`, which handles:
- Model training
- Accuracy computation
- Confusion matrix generation (for classifiers)
- Performance reporting (classification or regression metrics)

### 🧠 Dual Model Support
Out-of-the-box support for both:
- **Classification models** such as Logistic Regression, SVM, Random Forest, XGBoost
- **Regression models** such as Linear Regression, Ridge, SVR, Gradient Boosting

### 📊 Visual Performance Reports
- Heatmaps of confusion matrices
- Auto-generated `classification_report` with precision, recall, F1-score
- Regression summary with metrics like MAE, MSE, R²
- Tabular performance summary across all models

### 📁 Custom Dataset Integration
- Use default Scikit-Learn datasets or plug in your own CSV
- Built-in support for label encoding and numeric feature conversion
- Easy integration with Pandas for pre-processing pipelines

### 🧠 Deep Learning Ready
- Expandable with PyTorch and TensorFlow architectures
- Importable modules for CNNs, RNNs, and Transformers

### 🧪 Educational & Research Utility
- Ideal for teaching ML fundamentals in a comparative format
- Benchmarking for internal ML pipelines and research reproducibility

---

## 🧠 Classification Models

| Model                  | Module                        |
|------------------------|-------------------------------|
| Logistic Regression    | `sklearn.linear_model`        |
| Support Vector Machine | `sklearn.svm`                 |
| Decision Tree          | `sklearn.tree`                |
| Random Forest          | `sklearn.ensemble`            |
| k-Nearest Neighbors    | `sklearn.neighbors`           |
| Gaussian Naive Bayes   | `sklearn.naive_bayes`         |
| XGBoost Classifier     | `xgboost.XGBClassifier`       |

---

## 📉 Regression Models

| Model                        | Module                              |
|-----------------------------|--------------------------------------|
| Linear Regression           | `sklearn.linear_model.LinearRegression` |
| Ridge Regression            | `sklearn.linear_model.Ridge`        |
| Support Vector Regressor    | `sklearn.svm.SVR`                   |
| Decision Tree Regressor     | `sklearn.tree.DecisionTreeRegressor` |
| Random Forest Regressor     | `sklearn.ensemble.RandomForestRegressor` |
| Gradient Boosting Regressor | `sklearn.ensemble.GradientBoostingRegressor` |
| k-NN Regressor              | `sklearn.neighbors.KNeighborsRegressor` |

___

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

---

## 🧪 How to Run

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook balances.ipynb
```

---

### 📁 Customize Dataset

Replace dataset ingestion cell with:

```python
import pandas as pd
df = pd.read_csv("your_dataset.csv")
X = df.drop("target_column", axis=1)
y = df["target_column"]
```

---

### 📊 Outputs

- R², MAE, MSE for each model
- Bar plots of performance scores
- Visual predicted vs. actual scatter charts
- Residual error analysis

---

## 🔮 Roadmap

- [ ] Add time series models (Prophet, ARIMA)
- [ ] Integrate GridSearchCV for model tuning
- [ ] SHAP-based interpretability
- [ ] Flask/FastAPI API for deploying forecasts
- [ ] LLM summarization of forecast outcomes

---

## 🤝 Contributing

1. 🍴 Fork the project
2. 🔧 Create a branch: `git checkout -b feat/new-feature`
3. ✅ Commit and push changes
4. 📬 Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License**.

---

