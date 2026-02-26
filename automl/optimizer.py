import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np


def optimize_model(model, preprocessor, X, y, problem_type):

    def objective(trial):

        # 🔥 Tune Random Forest
        if "RandomForest" in model.__class__.__name__:
            model.n_estimators = trial.suggest_int("n_estimators", 50, 200)
            model.max_depth = trial.suggest_int("max_depth", 3, 15)

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        if problem_type == "classification":

            unique, counts = np.unique(y, return_counts=True)
            min_class_samples = counts.min()

            # 🔥 If any class has <2 samples → disable CV
            if min_class_samples < 2:
                pipeline.fit(X, y)
                preds = pipeline.predict(X)
                return accuracy_score(y, preds)

            cv = min(3, min_class_samples)

            score = cross_val_score(
                pipeline,
                X,
                y,
                cv=cv,
                scoring="accuracy"
            ).mean()

        else:
            score = cross_val_score(
                pipeline,
                X,
                y,
                cv=3,
                scoring="r2"
            ).mean()

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    return study.best_value, study.best_params