import joblib
import os

MODEL_PATH = "model_registry"

os.makedirs(MODEL_PATH, exist_ok=True)


def save_registered_model(model, model_name, feature_columns):
    """
    Save model together with training schema
    """

    model_file = os.path.join(MODEL_PATH, "best_model.pkl")

    payload = {
        "model": model,
        "features": feature_columns
    }

    joblib.dump(payload, model_file)


def load_latest_model():
    """
    Load model and training schema
    """

    model_file = os.path.join(MODEL_PATH, "best_model.pkl")

    payload = joblib.load(model_file)

    return payload["model"], payload["features"]