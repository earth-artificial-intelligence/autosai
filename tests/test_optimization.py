import pytest
import numpy as np
from autosai.optimization.bayesian_optimization import BayesianOptimization
from autosai.model_selection import ModelSelector

def test_bayesian_optimization():
    selector = ModelSelector()
    optimizer = BayesianOptimization(selector.create_model)

    X_train = np.random.rand(100, 10)
    y_train = np.random.rand(100, 1)
    X_val = np.random.rand(20, 10)
    y_val = np.random.rand(20, 1)

    best_params = optimizer.optimize(X_train, y_train, X_val, y_val, n_trials=2)
    assert 'model_type' in best_params
