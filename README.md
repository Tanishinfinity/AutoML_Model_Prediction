# 🧠 Mini AutoML Platform

An automated machine learning system that allows users to upload a dataset and automatically:

- Detect problem type (Classification / Regression)
- Preprocess data
- Train multiple ML models
- Perform Bayesian Optimization
- Compare model performance
- Select and save the best model
- Visualize performance and feature importance

---

## 🚀 Features

- Automated preprocessing pipeline
- Smart hyperparameter tuning (Optuna)
- Model comparison dashboard
- Performance visualization
- Feature importance graph
- Robust handling of small datasets
- Model persistence (.pkl)

---

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- Optuna
- XGBoost
- Pandas / NumPy
- Matplotlib

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py