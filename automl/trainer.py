from automl.model_space import get_models
from automl.optimizer import optimize_model
from sklearn.pipeline import Pipeline


def train_all_models(problem_type, preprocessor, X_train, y_train):
    models = get_models(problem_type)

    results = {}

    for name, model in models.items():
        score, params = optimize_model(
            model,
            preprocessor,
            X_train,
            y_train,
            problem_type
        )

        results[name] = {
            "score": score,
            "params": params
        }

    return results


def train_final_model(best_model, preprocessor, X, y):
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", best_model)
    ])

    pipeline.fit(X, y)
    return pipeline