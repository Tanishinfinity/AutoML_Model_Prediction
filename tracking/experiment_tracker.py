import pandas as pd
import os

LOG_FILE = "experiments.csv"


def log_experiment(model_name, score):

    data = {
        "model": [model_name],
        "score": [score]
    }

    df = pd.DataFrame(data)

    if os.path.exists(LOG_FILE):

        df.to_csv(LOG_FILE, mode="a", header=False, index=False)

    else:

        df.to_csv(LOG_FILE, index=False)