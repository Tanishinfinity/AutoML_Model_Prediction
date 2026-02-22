import joblib

def save_model(model, filename="best_model.pkl"):
    joblib.dump(model, filename)

def load_model(filename="best_model.pkl"):
    return joblib.load(filename)