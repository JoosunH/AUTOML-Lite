# 🤖 AutoML Lite

A lightweight, interpretable AutoML pipeline for tabular datasets.

This project enables users to upload CSV files and automatically preprocess data, detect task type (classification or regression), select the best model using Optuna hyperparameter tuning, evaluate model performance, and download the final ML pipeline — all via an interactive Streamlit UI.

---

## 📌 Features

- ✅ Task Type Detection (Classification vs Regression)
- 🧹 Data Preprocessing with ColumnTransformer
- 🔍 Hyperparameter Tuning using **Bayesian Optimization** (Optuna)
- 📈 Evaluation Metrics (Accuracy, Precision, Recall, ROC-AUC, MAE, RMSE, R^2)
- 📊 Visualizations: Confusion Matrix, ROC Curve, Residual Plots, Leaderboard
- 💾 One-click model pipeline export (.joblib)
- 🧪 Optional experiment tracking via **MLflow**

---

## 🖥️ App Preview

<img src="assets/screenshot_ui.png" width="500" />

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/yourusername/automl-lite.git
cd automl-lite

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run UI/Front.py
```

---

## 📄 Research Paper

This project is documented in a technical paper that explains the motivation, architecture, and evaluation of AutoML Lite, including its use of **Bayesian optimization**, task detection, and lightweight design principles.

👉 [Read the paper here](./AutoML_Research_Paper.pdf)

---

## 🧠 Datasets Used

- Titanic Dataset (Classification)
- Dropout Prediction (Multiclass Classification)
- Football Transfer Fee Prediction (Regression)

---

## 🛠️ Technologies

- **Languages**: Python
- **ML Libraries**: scikit-learn, XGBoost, Optuna, Keras, Pandas, NumPy
- **Visualization**: matplotlib, seaborn
- **App/UI**: Streamlit
- **Deployment**: Local or cloud via Streamlit sharing

---

## 📚 References

- Optuna: https://optuna.org/
- scikit-learn: https://scikit-learn.org/
- Streamlit: https://streamlit.io/
- XGBoost: https://xgboost.ai/
- [Bayesian Optimization Book](https://bayesoptbook.com/book/bayesoptbook.pdf)

---

## 👨‍💻 Author

**Joosun Hwang**  
📍 Toronto, Canada  
💼 [LinkedIn](https://linkedin.com/in/joosun-hwang-931971234/) | 💻 [GitHub](https://github.com/JoosunH)

---

## 📢 License

MIT License
