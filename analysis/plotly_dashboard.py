import plotly.express as px


def leaderboard_chart(df):

    fig = px.bar(
        df,
        x="Model",
        y="Score",
        color="Model",
        title="Model Leaderboard"
    )

    return fig


def correlation_plot(df):

    numeric_df = df.select_dtypes(include=["number"])

    fig = px.imshow(
        numeric_df.corr(),
        title="Correlation Heatmap"
    )

    return fig