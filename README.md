# 🧠 AutoML Model Prediction Platform

An end-to-end Automated Machine Learning (AutoML) system built with Python and Streamlit that automatically trains, optimizes, compares, and selects the best machine learning model for a given dataset.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Overview

The **AutoML Model Prediction Platform** allows users to upload a CSV dataset and automatically:

- Detect problem type (Classification or Regression)
- Perform intelligent data preprocessing
- Train multiple ML models
- Apply Bayesian Hyperparameter Optimization using Optuna
- Compare model performance
- Select the best-performing model
- Visualize performance with charts
- Save the trained model as `.pkl`

This project demonstrates practical implementation of real-world AutoML systems used in modern AI workflows.

---

## 🏗️ System Architecture

```
User Upload
      ↓
Data Preprocessing
      ↓
Model Training
      ↓
Bayesian Optimization
      ↓
Model Comparison
      ↓
Best Model Selection
      ↓
Visualization + Model Saving
```

---

## 🚀 Key Features

- Automated ML pipeline
- Smart preprocessing engine (Scaling + Encoding)
- Dynamic handling of small & imbalanced datasets
- Bayesian hyperparameter tuning
- Multi-model comparison (Logistic Regression, Random Forest, XGBoost)
- Performance visualization (Bar Charts)
- Feature importance visualization (Tree-based models)
- Modular, production-style architecture
- Model persistence using Joblib
- Robust error handling

---

## 🛠 Tech Stack

- Python 3.11  
- Streamlit  
- Scikit-learn  
- Optuna  
- XGBoost  
- Pandas  
- NumPy  
- Matplotlib  
- Joblib  

---

## 📂 Project Structure

```
AutoML_Model_Prediction/
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
│
└── automl/
    ├── __init__.py
    ├── data_loader.py
    ├── preprocessing.py
    ├── model_space.py
    ├── optimizer.py
    ├── trainer.py
    ├── evaluator.py
    └── model_utils.py
```

---

## ▶️ Installation & Run

### 1️⃣ Clone Repository

```
git clone https://github.com/Tanishinfinity/AutoML_Model_Prediction.git
cd AutoML_Model_Prediction
```

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 3️⃣ Run Application

```
streamlit run app.py
```

---

## 📊 Workflow

1. Upload CSV dataset  
2. Select target column  
3. AutoML detects problem type  
4. Preprocessing pipeline is built automatically  
5. Multiple models are trained  
6. Bayesian optimization tunes hyperparameters  
7. Best model selected automatically  
8. Performance visualized  
9. Model saved for future predictions  

---

## 📈 Example Use Cases

- Sentiment classification
- Customer churn prediction
- Sales forecasting
- Employee performance analysis
- Academic ML experiments
- Business data analytics

---

## 🧠 What This Project Demonstrates

- End-to-end ML pipeline engineering
- Automated model selection
- Hyperparameter optimization using Bayesian methods
- Production-style modular architecture
- Robust dataset validation & error handling
- Explainable ML via feature importance visualization

---

## 🔮 Future Improvements

- SHAP Explainability integration
- Automatic NLP support (TF-IDF for text columns)
- Confusion matrix visualization
- REST API integration (FastAPI)
- Deployment on Streamlit Cloud
- Docker containerization

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Tanish**  
Machine Learning Enthusiast | AI Developer  

⭐ If you found this project useful, consider giving it a star!