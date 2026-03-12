# 🧠 AutoML Model Prediction Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-orange?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Machine-Learning-brightgreen?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end **Automated Machine Learning (AutoML)** system built with **Python and Streamlit**. This platform automates the most complex steps of the machine learning workflow, including preprocessing, problem detection, model selection, hyperparameter tuning, and deployment-ready prediction pipelines.

---

## 📌 Project Overview

The **AutoML Model Prediction Platform** simplifies the data science lifecycle. Users upload a raw dataset, and the system automatically performs validation, cleaning, feature engineering, and optimization. 

The platform supports both **Classification and Regression** tasks, providing a high-level interface for users to build powerful models without writing a single line of code.

---

## 🚀 Key Features

* **End-to-End Automation:** Seamless transition from raw data to a trained model.
* **Problem Type Detection:** Automatically identifies if the task is classification or regression.
* **Intelligent Preprocessing:** Automatic handling of missing values, encoding categorical variables, and label normalization.
* **Bayesian Optimization:** Uses **Optuna** for efficient hyperparameter tuning compared to traditional GridSearch.
* **Model Leaderboard:** Trains and compares multiple models (XGBoost, Random Forest, etc.) to select the best performer.
* **Feature Importance:** Built-in visualizations to understand which variables drive your model's decisions.
* **Schema Alignment:** Ensures that new prediction data matches the training data format, preventing runtime errors.
* **Model Persistence:** Save and reload your best-performing models using **Joblib**.

---

## 🏗 System Workflow

### 1. Training Pipeline
`Dataset Upload` → `Validation` → `Automatic Cleaning` → `Feature Engineering` → `Problem Detection` → `Model Training` → `Hyperparameter Tuning (Optuna)` → `Model Comparison` → `Best Model Selection` → `Insights Visualization`

### 2. Prediction Pipeline
`Upload Prediction CSV` → `Schema Alignment` → `Apply Stored Preprocessing` → `Run Trained Model` → `Display Results`

---

## 🛠 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| **Python 3.11** | Core logic and programming |
| **Streamlit** | Interactive web dashboard |
| **Scikit-learn** | ML models and preprocessing pipelines |
| **Optuna** | Bayesian hyperparameter optimization |
| **XGBoost** | High-performance gradient boosting |
| **Pandas / NumPy** | Data manipulation and processing |
| **Matplotlib** | Model performance visualizations |
| **Joblib** | Model and pipeline serialization |

---

## 📂 Project Structure

```text
AutoML_Model_Prediction/
│
├── app.py                      # Streamlit UI & main entry point
├── requirements.txt            # Project dependencies
├── README.md                   # Documentation
├── LICENSE                     # MIT License
│
└── automl/                     # Core AutoML Logic
    ├── __init__.py
    ├── data_loader.py          # Data validation & ingestion
    ├── preprocessing.py        # Automated cleaning pipelines
    ├── feature_engineering.py  # Feature transformation logic
    ├── model_space.py          # Model definitions & search spaces
    ├── optimizer.py            # Optuna tuning logic
    ├── trainer.py              # Training & comparison engine
    ├── evaluator.py            # Performance metrics & plots
    └── model_utils.py          # Persistence (Save/Load) logic
▶️ Installation & Setup
1. Clone the repository
Bash
git clone [https://github.com/Tanishinfinity/AutoML_Model_Prediction.git](https://github.com/Tanishinfinity/AutoML_Model_Prediction.git)
cd AutoML_Model_Prediction
2. Install dependencies
Bash
pip install -r requirements.txt
3. Run the application
Bash
streamlit run app.py
Access the platform in your browser at: http://localhost:8501

📈 Example Use Cases
Business: Customer churn prediction and sales forecasting.

Marketing: Sentiment analysis and engagement classification.

Research: Rapid prototyping for academic machine learning experiments.

Startups: Fast-tracking the MVP development for AI features.

🔮 Future Improvements
[ ] SHAP Explainability: Detailed instance-level model interpretation.

[ ] NLP Pipelines: Support for raw text columns and automated vectorization.

[ ] API Layer: Automated generation of FastAPI endpoints for models.

[ ] Dockerization: Easy deployment via containerization.

📜 License
This project is licensed under the MIT License.

👨‍💻 Author
Tanish Machine Learning Enthusiast | AI Developer

⭐ If you find this project useful, please consider giving it a star on GitHub!