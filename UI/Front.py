# ui/main.py

import streamlit as st
import pandas as pd
import csv
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
    precision_score,
    recall_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
# Add root path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from sklearn.model_selection import train_test_split
from app.data_preprocessing import preprocess_data
from app.model_selector import get_models
from app.tuner import tune_model, load_csv_auto_detect
from app.evaluator import evaluate_model
from app.pipeline import build_and_save_pipeline
from app.visuals import plot_target_distribution, plot_model_leaderboard


st.set_page_config(page_title="AutoML Lite", layout="wide")

st.title("🤖 AutoML Lite App")
st.write("Upload a CSV, pick the target column, and we'll train + tune models for you.")

uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file:
    df = load_csv_auto_detect(uploaded_file)
    st.success("✅ Data uploaded successfully!")

    st.subheader("📌 Preview of Dataset")
    st.dataframe(df.head())

    st.subheader("🎯 Choose Target Column")
    classification_candidates = []
    regression_candidates = []

    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        dtype = df[col].dtype
        col_lower = col.lower()

        # 🚫 Skip constant or ID-like columns
        if nunique <= 1 or nunique >= len(df) * 0.9:
            continue
        if any(term in col_lower for term in ['id', 'uuid', 'index']):
            continue

        # ✅ Classifier candidates
        if nunique <= 10 and dtype.kind in "iOcbf":
            classification_candidates.append((col, nunique))
        # ✅ Regressor candidates
        elif dtype.kind in "ifc" and nunique > 20:
            regression_candidates.append((col, nunique))

    classification_candidates.sort(key=lambda x: x[1])
    regression_candidates.sort(key=lambda x: x[1], reverse=True)

    recommendations = []

    if classification_candidates:
        col, n = classification_candidates[0]
        recommendations.append(f"🎯 {col} (classification — {n} classes)")
    if regression_candidates:
        col, n = regression_candidates[0]
        recommendations.append(f"📈 {col} (regression — {n} unique values)")

    if recommendations:
        st.markdown("✅ **Suggested target columns based on data shape:**")
        for rec in recommendations:
            st.markdown(f"- {rec}")
    else:
        st.info("ℹ️ No specific recommendations available. Please choose a target column manually.")

    target_col = st.selectbox("Select Target Column", df.columns, key="target_col_main")

    if st.button("🚀 Run AutoML"):
        with st.spinner("Running full pipeline..."):
            
            if target_col not in df.columns:
                st.error("❌ Target column not found in dataset.")
                st.stop()
            if df[target_col].isnull().any():
                st.error("❌ Target column contains null values. Please clean your data.")
                st.stop()
            
            # Step 1: Preprocessing
            preprocessor, X_clean, y, task_type = preprocess_data(df, target_col)
            st.success(f"✅ Preprocessing complete. Task type: {task_type}")
            st.subheader("🎯 Target Distribution")
            fig_target = plot_target_distribution(y)
            st.pyplot(fig_target)
            
            X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2, random_state=42)

            # Step 2: Model Selection
            models = get_models(task_type)

            # Step 3: Tuning
            results = []
            for name, model in models.items():
                st.write(f"🔧 Tuning {name}...")
                best_model, score, params = tune_model(name, model, X_train, y_train, task_type, n_trials=5)
                
                try:
                    best_model.fit(X_train, y_train)
                except ValueError as e:
                    st.error(f"🚨 Model training failed for {name}: {str(e)}")
                    continue 
                
                
                results.append({
                    "name": name,
                    "model": best_model,
                    "score": score,
                    "params": params
                })
            if not results:
                st.error("❌ All models failed during training.")
                st.stop()
            # Step 4: Evaluate & Plot
            final_model, final_score, final_name = evaluate_model(results)
            
            y_pred = final_model.predict(X_test)
            st.subheader("📉 Model Evaluation Metrics")
            
            if task_type == "classification":
                cm = confusion_matrix(y_test, y_pred)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                fig, ax = plt.subplots()
                disp.plot(ax=ax)
                st.pyplot(fig)
                precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            
                try:
                    if len(set(y_test)) == 2:
                            y_proba = final_model.predict_proba(X_clean)[:, 1]
                            auc = roc_auc_score(y_test, y_proba)
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            fig_auc, ax_auc = plt.subplots()
                            ax_auc.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
                            ax_auc.plot([0, 1], [0, 1], "k--")
                            ax_auc.set_xlabel("False Positive Rate")
                            ax_auc.set_ylabel("True Positive Rate")
                            ax_auc.set_title("ROC Curve")
                            ax_auc.legend()
                            st.pyplot(fig_auc)
                except Exception:
                    st.warning("⚠️ ROC-AUC could not be computed.")

                st.markdown(f"✅ **Precision:** {precision:.3f}  |  **Recall:** {recall:.3f}")
        
            elif task_type == "regression":
                mae = mean_absolute_error(y_test, y_pred)
                rmse = root_mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                st.markdown(f"📈 **MAE:** {mae:.3f}  |  **RMSE:** {rmse:.3f}  |  **R²:** {r2:.3f}")

                # Residual plot
                fig_resid, ax_resid = plt.subplots()
                ax_resid.scatter(y_pred, y_test - y_pred, alpha=0.5)
                ax_resid.axhline(0, color="red", linestyle="--")
                ax_resid.set_xlabel("Predicted Values")
                ax_resid.set_ylabel("Residuals")
                ax_resid.set_title("Residual Plot")
                st.pyplot(fig_resid)
            st.subheader("📋 Model Score Summary")
            leaderboard_df = plot_model_leaderboard(results)
            st.dataframe(leaderboard_df)
    
            # Step 5: Save Full Pipeline
            X_raw = df.drop(columns=[target_col])
            X_train_raw, X_test_raw = train_test_split(X_raw, test_size=0.2, random_state=42)
            filename = f"{final_name.replace(' ', '_')}_pipeline.joblib"
            full_pipeline = build_and_save_pipeline(preprocessor, final_model, X_train_raw, y_train, filename)

            st.success(f"🎯 Best model: {final_name} (Score: {final_score:.4f})")
            with open(f"models/{filename}", "rb") as f:
                st.download_button("💾 Download Model Pipeline", f, file_name=filename)


def load_csv_auto_detect(file):
    # Try to detect delimiter
    sample = file.read(1024).decode("utf-8")
    file.seek(0)  # reset pointer
    try:
        dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(file, delimiter=dialect.delimiter)
    except csv.Error:
        # fallback to comma if detection fails
        df = pd.read_csv(file)
    return df