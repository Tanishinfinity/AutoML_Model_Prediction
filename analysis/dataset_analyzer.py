import pandas as pd

def analyze_dataset(df):

    report = {}

    report["Rows"] = df.shape[0]
    report["Columns"] = df.shape[1]
    report["Missing Values"] = df.isnull().sum().sum()
    report["Duplicate Rows"] = df.duplicated().sum()

    report["Numeric Features"] = list(df.select_dtypes(include=["number"]).columns)
    report["Categorical Features"] = list(df.select_dtypes(include=["object"]).columns)

    return report