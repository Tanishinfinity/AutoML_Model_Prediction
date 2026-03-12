from sklearn.ensemble import VotingClassifier, VotingRegressor


def build_ensemble(models, problem_type):

    estimators = [(name, model) for name, model in models.items()]

    if problem_type == "classification":

        ensemble = VotingClassifier(
            estimators=estimators,
            voting="soft"
        )

    else:

        ensemble = VotingRegressor(
            estimators=estimators
        )

    return ensemble