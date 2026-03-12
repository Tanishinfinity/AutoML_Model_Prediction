from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import pandas as pd


# ----------------------------------------------------
# Detect Problem Type
# ----------------------------------------------------
def detect_problem_type(target_series):

    if target_series.dtype == "object":
        return "classification"

    if target_series.nunique() <= 20:
        return "classification"

    return "regression"


# ----------------------------------------------------
# Clean Target Labels
# ----------------------------------------------------
def clean_target_labels(y):

    y = y.astype(str)

    # lowercase
    y = y.str.lower()

    # remove extra spaces
    y = y.str.strip()

    # remove multiple spaces
    y = y.str.replace(r"\s+", " ", regex=True)

    # convert invalid text to NaN
    y = y.replace(["nan", "none", "missing"], pd.NA)

    return y


# ----------------------------------------------------
# Preprocess Data
# ----------------------------------------------------
def preprocess_data(df, target_col):

    # remove rows where target is missing
    df = df.dropna(subset=[target_col])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = detect_problem_type(y)

    # ----------------------------------------------------
    # CLEAN TARGET
    # ----------------------------------------------------
    if problem_type == "classification":

        y = clean_target_labels(y)

        # remove rows where cleaning created NA
        valid_rows = y.notna()

        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        # ----------------------------------------------------
        # ENCODE TARGET LABELS
        # ----------------------------------------------------
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # ----------------------------------------------------
    # FEATURE TYPES
    # ----------------------------------------------------
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    # ----------------------------------------------------
    # NUMERIC PIPELINE
    # ----------------------------------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # ----------------------------------------------------
    # CATEGORICAL PIPELINE
    # ----------------------------------------------------
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # ----------------------------------------------------
    # COLUMN TRANSFORMER
    # ----------------------------------------------------
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # ----------------------------------------------------
    # TRAIN TEST SPLIT
    # ----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return preprocessor, X_train, X_test, y_train, y_test