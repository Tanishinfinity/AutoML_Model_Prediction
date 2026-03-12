from sklearn.pipeline import Pipeline
from automl.model_space import get_models
from automl.optimizer import optimize_model


def train_all_models(problem_type, preprocessor, X_train, y_train):

    models = get_models(problem_type)

    results = {}

    for model_name, model in models.items():

        score, params, study = optimize_model(
            model,
            preprocessor,
            X_train,
            y_train,
            problem_type
        )

        results[model_name] = {
            "score": score,
            "params": params,
            "study": study
        }

    return results


def train_final_model(model, preprocessor, X_train, y_train):

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)

    return pipeline