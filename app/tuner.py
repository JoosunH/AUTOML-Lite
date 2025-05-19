import optuna
from sklearn.model_selection import cross_val_score
import pandas as pd
import csv

def tune_model(model_name, model, X, y, task_type, n_trials=20):
    '''
    Tune a single model using Optuna and return the best estimator and score.

    '''
    
    def objective(trial):
        params = {}
        
        if model_name == 'Random Forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 5)
        elif model_name == 'XGBoost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 15)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3)
        elif model_name == 'Logistic Regression':
            params['C'] = trial.suggest_float('C', 0.01, 10.0)
            #Think of C as how much you're trusting your training data:
            #A small C says: “Be conservative. Don’t trust this data too much.”, strong regularization strength also Simpler model, avoids overfitting
            #A large C says: “Fit this data as closely as possible, even if it overfits.” complex model, more prone to overfitting
        elif model_name == 'KNN':
            params['n_neighbors'] = trial.suggest_int('n_neighbors', 3, 15)
        elif model_name == 'SVM':
            params['C'] = trial.suggest_float('C', 0.01, 10.0)
            params['kernel'] = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        elif model_name == 'Lasso Regression':
            params['alpha'] = trial.suggest_float('alpha',  0.01, 1.0, log=True)
        elif model_name == 'Ridge Regression':
            params['alpha'] = trial.suggest_float('alpha', 0.01, 1.0, log=True)
        elif model_name == 'SVR':
            params['C'] = trial.suggest_float('C', 0.01, 10.0)    
            #kernel is a function that transforms your data into a higher-dimensional space so the algorithm can find a better decision boundary. linear is a straight line, rbf is a curve.
            '''
            Use linear if:
            The data is roughly linearly separable.
            You want fast training.
            You have a lot of features (like text data, TF-IDF vectors).

            Use rbf if:
                The relationship between features and labels is non-linear.
                You want the model to discover complex patterns.'''
                
        model.set_params(**params)
        #Applying the trial suggested parameters to the model
        score = cross_val_score(model, X, y, cv=3, scoring='accuracy' if task_type == 'classification' else 'r2').mean()
        #Within new parameters, we are using cross-validation to evaluate the model's performance, and this happens in every trial.
        return score
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    model.set_params(**study.best_params)
    model.fit(X, y)
    
    #best_value is the best score achieved during the optimization process from beginning to end
    #Its not showing the best value of model, it just shows the best score for the best parameters
    return model, study.best_value, study.best_params



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