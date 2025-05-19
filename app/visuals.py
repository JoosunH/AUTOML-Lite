import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_model_leaderboard(results):
    df = pd.DataFrame([
        {
            "Model": r["name"],
            "Score": round(r["score"], 4),
            **r["params"]
        }
        for r in results
    ])
    return df

def plot_target_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 4))
    if y.nunique() <= 20:
        y.value_counts().plot(kind="bar", ax=ax)
        ax.set_title("Target Class Distribution")
    else:
        sns.histplot(y, bins=20, ax=ax)
        ax.set_title("Target Value Distribution (Regression)")
    return fig