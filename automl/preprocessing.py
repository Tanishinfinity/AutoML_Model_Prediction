from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def detect_problem_type(target):
    if target.dtype == 'object':
        return "classification"
    else:
        return "regression"


def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 🔥 Clean text labels (optional but recommended)
    if y.dtype == 'object':
        y = y.astype(str).str.lower().str.strip()

    # 🔥 Encode target for classification
    label_encoder = None
    if y.dtype == 'object':
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

    # Detect feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    transformers = []

    if len(numerical_cols) > 0:
        transformers.append(("num", StandardScaler(), numerical_cols))

    if len(categorical_cols) > 0:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    preprocessor = ColumnTransformer(transformers=transformers)

    # 🔥 Safe stratification
    stratify_option = None
    if len(set(y)) > 1:
        unique, counts = pd.Series(y).value_counts().index, pd.Series(y).value_counts().values
        if counts.min() >= 2:
            stratify_option = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_option
    )

    return preprocessor, X_train, X_test, y_train, y_test 