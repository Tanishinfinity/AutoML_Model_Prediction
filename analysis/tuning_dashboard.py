import pandas as pd
import matplotlib.pyplot as plt


def tuning_history(study):

    trials = study.trials_dataframe()

    fig, ax = plt.subplots()

    ax.plot(trials["value"])

    ax.set_title("Hyperparameter Optimization Progress")
    ax.set_xlabel("Trial")
    ax.set_ylabel("Score")

    return fig


def parameter_importance(study):

    trials = study.trials_dataframe()

    params = trials.filter(like="params")

    importance = params.var().sort_values(ascending=False)

    fig, ax = plt.subplots()

    importance.plot(kind="bar", ax=ax)

    ax.set_title("Parameter Importance")

    return fig