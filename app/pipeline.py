import joblib
from sklearn.pipeline import Pipeline

def build_and_save_pipeline(preprocessor, model, X_raw, y, filename="best_model_pipeline.joblib"):
    """
    Builds a full pipeline (preprocessor + model), fits it, and saves to disk.

    Parameters:
    - preprocessor: ColumnTransformer from preprocessing step
    - model: Best trained model from evaluation
    - X_raw: Original, unprocessed features (pandas DataFrame)
    - y: Target column
    - filename: File name to save the pipeline to
    """
    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    pipeline.fit(X_raw, y)
    # Save the pipeline to a file
    joblib.dump(pipeline, f"models/{filename}")
    print(f"✅ Full pipeline saved as: models/{filename}")
    return pipeline