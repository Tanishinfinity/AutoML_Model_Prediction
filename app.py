import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from automl.data_loader import load_data
from automl.preprocessing import detect_problem_type, preprocess_data
from automl.trainer import train_all_models, train_final_model
from automl.evaluator import select_best_model
from automl.model_space import get_models

from registry.model_registry import save_registered_model, load_latest_model
from tracking.experiment_tracker import log_experiment


# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="AutoML Model Prediction Platform",
    layout="wide"
)

st.title("🧠 AutoML Model Prediction Platform")
st.write("Upload a dataset and automatically train machine learning models.")


# =================================================
# TRAINING DATASET
# =================================================

uploaded_file = st.file_uploader("Upload Training Dataset", type=["csv"])

if uploaded_file:

    df = load_data(uploaded_file)

    st.success("Dataset loaded successfully")

    rows, cols = df.shape
    missing = df.isna().sum().sum()

    c1, c2, c3 = st.columns(3)

    c1.metric("Rows", rows)
    c2.metric("Columns", cols)
    c3.metric("Missing Values", missing)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())


# =================================================
# AUTOML TRAINING
# =================================================

    st.subheader("Train AutoML Model")

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("Run AutoML"):

        problem_type = detect_problem_type(df[target])

        st.success(f"Detected Problem Type: {problem_type}")

        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(
            df,
            target
        )

        with st.spinner("Training models..."):

            results = train_all_models(
                problem_type,
                preprocessor,
                X_train,
                y_train
            )


# =================================================
# MODEL LEADERBOARD
# =================================================

        leaderboard = pd.DataFrame([
            {"Model": m, "Score": info["score"]}
            for m, info in results.items()
        ])

        st.subheader("Model Leaderboard")

        st.dataframe(leaderboard.sort_values("Score", ascending=False))

        fig = px.bar(
            leaderboard,
            x="Model",
            y="Score",
            color="Model"
        )

        fig.update_layout(template="plotly_dark")

        st.plotly_chart(fig, use_container_width=True)


# =================================================
# BEST MODEL
# =================================================

        best_model_name, best_info = select_best_model(results)

        st.success(f"Best Model: {best_model_name}")

        params_df = pd.DataFrame(
            list(best_info["params"].items()),
            columns=["Parameter", "Value"]
        )

        st.subheader("Recommended Hyperparameters")
        st.dataframe(params_df)


# =================================================
# TRAIN FINAL MODEL
# =================================================

        models = get_models(problem_type)

        best_model = models[best_model_name]

        best_model.set_params(**best_info["params"])

        pipeline = train_final_model(
            best_model,
            preprocessor,
            X_train,
            y_train
        )

        save_registered_model(
            pipeline,
            best_model_name,
            list(X_train.columns)
        )

        log_experiment(best_model_name, best_info["score"])


# =================================================
# FEATURE IMPORTANCE
# =================================================

        st.subheader("Feature Importance")

        try:

            if hasattr(best_model, "feature_importances_"):

                feature_names = pipeline.named_steps["prep"].get_feature_names_out()

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": best_model.feature_importances_
                })

                importance_df = importance_df.sort_values(
                    "Importance",
                    ascending=False
                ).head(15)

                fig = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation="h"
                )

                fig.update_layout(template="plotly_dark")

                st.plotly_chart(fig, use_container_width=True)

        except:
            st.warning("Feature importance unavailable.")


# =================================================
# MODEL EVALUATION
# =================================================

        st.subheader("Model Evaluation")

        y_pred = pipeline.predict(X_test)

        if problem_type == "classification":

            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots()

            sns.heatmap(cm, annot=True, cmap="Blues", ax=ax)

            st.pyplot(fig)

            report = classification_report(
                y_test,
                y_pred,
                output_dict=True
            )

            st.dataframe(pd.DataFrame(report).transpose())

        else:

            c1, c2, c3 = st.columns(3)

            c1.metric("MAE", round(mean_absolute_error(y_test, y_pred), 3))
            c2.metric("MSE", round(mean_squared_error(y_test, y_pred), 3))
            c3.metric("R2", round(r2_score(y_test, y_pred), 3))


# =================================================
# PREDICTION TOOL
# =================================================

st.subheader("Predict New Data")

predict_file = st.file_uploader(
    "Upload dataset for prediction",
    type=["csv"],
    key="prediction"
)

if predict_file:

    model, feature_columns = load_latest_model()

    df_pred = pd.read_csv(predict_file)

    st.write("Prediction Dataset Preview")
    st.dataframe(df_pred.head())

    # ----------------------------------------------
    # ALIGN COLUMNS
    # ----------------------------------------------

    for col in feature_columns:
        if col not in df_pred.columns:
            df_pred[col] = None

    df_pred = df_pred[feature_columns]

    # ----------------------------------------------
    # CLEAN DATA USING PIPELINE INFO
    # ----------------------------------------------

    prep = model.named_steps["prep"]

    num_cols = prep.transformers_[0][2]
    cat_cols = prep.transformers_[1][2]

    for col in num_cols:
        if col in df_pred.columns:
            df_pred[col] = pd.to_numeric(df_pred[col], errors="coerce")
            df_pred[col] = df_pred[col].fillna(0)

    for col in cat_cols:
        if col in df_pred.columns:
            df_pred[col] = df_pred[col].astype(str)
            df_pred[col] = df_pred[col].replace(
                ["nan", "None", "<NA>"],
                "missing"
            )
            df_pred[col] = df_pred[col].fillna("missing")

    # ----------------------------------------------
    # PREDICT
    # ----------------------------------------------

    try:

        preds = model.predict(df_pred)

        df_pred["Prediction"] = preds

        st.success("Prediction completed")

        st.dataframe(df_pred)

        csv = df_pred.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:

        st.error("Prediction failed")

        st.write(e)