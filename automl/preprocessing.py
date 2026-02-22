from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def detect_problem_type(target):
    if target.dtype == 'object':
        return "classification"
    else:
        return "regression"


def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Detect column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    transformers = []

    # Add numerical transformer if exists
    if len(numerical_cols) > 0:
        transformers.append(
            ("num", StandardScaler(), numerical_cols)
        )

    # Add categorical transformer if exists
    if len(categorical_cols) > 0:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    # 🔥 Safe stratification handling
    stratify_option = None

    if y.dtype == 'object':
        class_counts = y.value_counts()

        # Only stratify if all classes have at least 2 samples
        if class_counts.min() >= 2:
            stratify_option = y

    # Perform train-test split safely
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_option
    )

    return preprocessor, X_train, X_test, y_train, y_test