import matplotlib.pyplot as plt
import seaborn as sns

def plot_missing_values(df):

    fig, ax = plt.subplots()

    df.isnull().sum().plot(kind="bar", ax=ax)

    ax.set_title("Missing Values per Column")

    return fig


def correlation_heatmap(df):

    numeric_df = df.select_dtypes(include=["number"])

    fig, ax = plt.subplots()

    sns.heatmap(numeric_df.corr(), cmap="coolwarm", ax=ax)

    ax.set_title("Correlation Heatmap")

    return fig