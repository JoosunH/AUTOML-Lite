import matplotlib.pyplot as plt


def evaluate_model(results, metrics='score', top_k = 3, verbose=True):
    """
    Evaluates and ranks all models based on their tuning score.
    
    Parameters:
    - results (list of dict): Each dict must contain 'name', 'score', 'model', 'params'
    - metric (str): Metric used for sorting (default: 'score')
    - top_k (int): How many top models to show
    - verbose (bool): Whether to print the leaderboard

    Returns:
    - best_model: The highest-scoring model
    - best_score: The score of the best model
    - best_name: Name of the best model
    """
    sorted_results = sorted(results, key=lambda x: x[metrics], reverse=True)
    
    if verbose:
        print("\n🏆 Model Leaderboard:")
        print("===================================")
        for i, result in enumerate(sorted_results[:top_k]):
            print(f"{i+1}. {result['name']} - {result['score']:.4f}")
            print(f"   Params: {result['params']}")
            print("===================================")
    
    best = sorted_results[0]
    return best['model'], best['score'], best['name']

