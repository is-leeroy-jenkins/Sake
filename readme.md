# ⚖️ Sake

**Sake** is your go-to, modular machine learning framework for Budget Execution data analysis built in Python with **Scikit**, **XGBoost**, **PyTorch**, and **TensorFlow**. Designed for rapid experimentation, visualization, and benchmarking of both **classification** and **regression** models, it provides a structured yet extensible workflow that’s equally useful for teaching, prototyping, and real-world application development.

---

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

---

## 📦 Dependencies

| Package          | Description                                                      | Link                                                  |
|------------------|------------------------------------------------------------------|-------------------------------------------------------|
| numpy            | Numerical computing library                                      | [numpy.org](https://numpy.org/)                      |
| pandas           | Data manipulation and DataFrames                                 | [pandas.pydata.org](https://pandas.pydata.org/)      |
| matplotlib       | Plotting and visualization                                       | [matplotlib.org](https://matplotlib.org/)            |
| seaborn          | Statistical graphics (built on top of matplotlib)               | [seaborn.pydata.org](https://seaborn.pydata.org/)    |
| scikit-learn     | Machine learning models and tools                                | [scikit-learn.org](https://scikit-learn.org/stable/) |
| xgboost          | Extreme Gradient Boosting library                                | [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) |
| torch            | PyTorch deep learning library                                    | [pytorch.org](https://pytorch.org/)                  |
| tensorflow       | Google’s ML framework                                            | [tensorflow.org](https://www.tensorflow.org/)        |
| openai           | OpenAI's API client for LLMs                                     | [openai-python](https://github.com/openai/openai-python) |
| requests         | HTTP requests for APIs                                           | [requests](https://requests.readthedocs.io/)         |
| PySimpleGUI      | Simplified GUI wrapper around Tkinter and Qt                    | [PySimpleGUI](https://pysimplegui.readthedocs.io/)   |
| typing           | Type hinting support (built-in for Python ≥3.5)                 | [typing](https://docs.python.org/3/library/typing.html) |
| pyodbc           | ODBC database access for SQL servers                             | [pyodbc](https://github.com/mkleehammer/pyodbc)      |
| fitz             | PDF and image rendering via PyMuPDF                             | [PyMuPDF](https://pymupdf.readthedocs.io/)           |
| pillow           | Image processing capabilities (PIL fork)                        | [pillow](https://python-pillow.org/)                 |
| openpyxl         | Excel file reader/writer                                         | [openpyxl](https://openpyxl.readthedocs.io/)         |
| soundfile        | Read/write sound files                                           | [soundfile](https://pysoundfile.readthedocs.io/)     |
| sounddevice      | Interface for sound playback and recording                       | [sounddevice](https://python-sounddevice.readthedocs.io/) |
| loguru           | Elegant logging library                                          | [loguru](https://github.com/Delgan/loguru)           |
| statsmodels      | Statistical modeling and hypothesis testing                      | [statsmodels](https://www.statsmodels.org/)          |
| dotenv           | Load environment variables from `.env` files                     | [dotenv](https://github.com/theskumar/python-dotenv) |
| python-dotenv    | Same as above (used for `.env` management)                      | [python-dotenv](https://saurabh-kumar.com/python-dotenv/) |

---

## 🧪 How to Run

### 🔧 Setup

```bash
git clone https://github.com/your-username/balance-projector.git
cd balance-projector
pip install -r requirements.txt
jupyter notebook models.ipynb
