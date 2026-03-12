from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def auto_feature_engineering(X, y, problem_type, k=10, poly=False):

    # Select best features
    if problem_type == "classification":
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
    else:
        selector = SelectKBest(score_func=f_regression, k=min(k, X.shape[1]))

    X_selected = selector.fit_transform(X, y)

    selected_features = selector.get_support(indices=True)

    # Optional polynomial features
    if poly:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        X_selected = poly.fit_transform(X_selected)

    return X_selected, selected_features