
import pandas as pd
from app.data_preprocessing import preprocess_data
from app.model_selector import get_models
from app.tuner import tune_model
from app.evaluator import evaluate_model
from app.pipeline import build_and_save_pipeline

# Load sample dataset
df = pd.read_csv("train.csv")

# Set your target column
target_col = "Survived"

# Step 1: Preprocess
preprocessing, X_clean, y, task_type = preprocess_data(df, target_col)
print(f"✅ Preprocessing complete! Task type detected: {task_type}")

# Step 2: Select models
models = get_models(task_type)
print(f"✅ {len(models)} model(s) selected: {list(models.keys())}\n")

all_results = []

# Step 3: Tune each model
for name, model in models.items():
    print(f"🔧 Tuning {name}...")
    best_model, best_score, best_params = tune_model(name, model, X_clean, y, task_type, n_trials=10)
    print(f"✅ {name} best score: {best_score:.4f}")
    print(f"📦 Best params: {best_params}\n")
    
    all_results.append({
        'name': name,
        'model': best_model,
        'score': best_score,
        'params': best_params
    })
    
final_model, final_score, final_name = evaluate_model(all_results)
print(f"\n🎯 Final selected model: {final_name} (Score: {final_score:.4f})")

X_raw = df.drop(columns=[target_col])
y = df[target_col]
filename = f"{final_name.replace(' ', '_').lower()}_pipeline.joblib"
full_pipeline = build_and_save_pipeline(preprocessing, final_model, X_raw, y, filename)



