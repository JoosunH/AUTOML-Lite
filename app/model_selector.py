from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

'''
-For XGBoost explanation of gradient boosting:
1)Start with a simple model (e.g., a small decision tree) that makes a rough prediction.

2)Calculate the errors (residuals) between the predicted and actual values.

3)Train a new model to predict these errors (the gradient of the loss function).

4)Add this new model to the overall prediction, improving the performance.

5)Repeat the process for many iterations.

Each new model “boosts” the performance of the entire ensemble by learning from the mistakes of the previous models — hence the name "boosting".

-Why gradient?
Because it uses the gradient of the loss function to figure out what the next model should focus on — similar to how gradient descent optimizes weights in neural networks.    
 
Gradient Descent: You have one model and you're adjusting its knobs (weights) gradually to reduce error.

Gradient Boosting: You keep adding small models (weak learners) that "nudge" your predictions in the right direction to reduce the error — like gradually sculpting a better model. 
    
    
'''



CLASSIFIERS = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(n_jobs=-1, random_state=42),
    'XGBoost': XGBClassifier(eval_metric='logloss'),
    #Evaluatoin metric logloss is commonly used for binary classification problems
    #It measures how close the predicted probabilities are to the actual class labels, Lower values indicate better performance.
    #XGboost uses gradient bo
    'KNN': KNeighborsClassifier(),
    'SVM' : SVC(probability=True)
}

REGRESSORS = {
    'Ridge Regression': Ridge(),
    # 'Lasso Regression': Lasso(max_iter=20000, tol=1e-4),
    'Random Forest': RandomForestRegressor(n_jobs=-1, random_state=42),
    'XGBoost': XGBRegressor(),
    'KNN': KNeighborsRegressor(),
    'SVR': SVR()
}

def get_models(task_type):
    task_type = task_type.lower()
    
    if task_type == 'classification':
        return CLASSIFIERS
    elif task_type == 'regression':
        return REGRESSORS
    else:
        raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")