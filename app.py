import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from automl.data_loader import load_data
from automl.preprocessing import detect_problem_type, preprocess_data
from automl.trainer import train_all_models, train_final_model
from automl.evaluator import select_best_model
from automl.model_space import get_models
from automl.model_utils import save_model


st.set_page_config(page_title="Mini AutoML", layout="wide")

st.title("🧠 Mini AutoML Platform")
st.write("Upload a dataset and automatically train & select the best model.")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    df = load_data(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Ensure dataset has at least 2 columns
    if df.shape[1] < 2:
        st.error("❌ Dataset must contain at least 2 columns (features + target).")
        st.stop()

    # Target selection
    target_col = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Run AutoML"):

        # Detect problem type
        problem_type = detect_problem_type(df[target_col])
        st.write(f"### 🔎 Detected Problem Type: `{problem_type}`")

        # Preprocess
        preprocessor, X_train, X_test, y_train, y_test = preprocess_data(df, target_col)

        # 🔥 Check feature existence
        if X_train.shape[1] == 0:
            st.error("❌ No feature columns available after selecting target column.")
            st.stop()

        # Train models
        with st.spinner("Training models... Please wait ⏳"):
            results = train_all_models(problem_type, preprocessor, X_train, y_train)

        # Convert results to DataFrame
        comparison_data = []
        for model_name, info in results.items():
            comparison_data.append({
                "Model": model_name,
                "Score": round(info["score"], 4)
            })

        comparison_df = pd.DataFrame(comparison_data)
        sorted_df = comparison_df.sort_values(by="Score", ascending=False)

        # Show table
        st.subheader("📊 Model Comparison")
        st.dataframe(sorted_df)

        # 📊 Performance Chart
        st.subheader("📈 Model Performance Chart")

        fig, ax = plt.subplots()
        ax.bar(sorted_df["Model"], sorted_df["Score"])
        ax.set_xlabel("Models")
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison")
        plt.xticks(rotation=45)

        st.pyplot(fig)

        # Select best model
        best_model_name, best_info = select_best_model(results)

        st.success(f"🏆 Best Model: {best_model_name}")
        st.write("Best Score:", round(best_info["score"], 4))
        st.write("Best Parameters:", best_info["params"])

        # Get model instance
        models = get_models(problem_type)
        best_model = models[best_model_name]

        # Apply best parameters
        best_model.set_params(**best_info["params"])

        # Train final pipeline
        final_pipeline = train_final_model(
            best_model,
            preprocessor,
            X_train,
            y_train
        )

        # Save model
        save_model(final_pipeline)
        st.success("💾 Model saved as 'best_model.pkl'")

        # 🔍 Feature Importance (safe)
        if hasattr(best_model, "feature_importances_"):

            st.subheader("🔍 Feature Importance")

            try:
                feature_names = final_pipeline.named_steps["prep"].get_feature_names_out()
                importances = best_model.feature_importances_

                importance_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances
                }).sort_values(by="Importance", ascending=False).head(15)

                fig2, ax2 = plt.subplots()
                ax2.barh(importance_df["Feature"], importance_df["Importance"])
                ax2.set_xlabel("Importance")
                ax2.set_title("Top Feature Importances")
                ax2.invert_yaxis()

                st.pyplot(fig2)

            except:
                st.info("Feature importance not available for this dataset.")