from .models import DenseModel, CNNModel, LSTMModel, TransformerModel, TabNetModel
from .optimization import BayesianOptimization

class ModelSelector:
    def __init__(self):
        self.models = {
            'dense': DenseModel,
            'cnn': CNNModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel,
            'tabnet': TabNetModel
        }

    def create_model(self, model_type, trial, input_shape):
        if model_type in self.models:
            return self.models[model_type](trial, input_shape)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, n_trials=50):
        optimizer = BayesianOptimization(self.create_model)
        return optimizer.optimize(X_train, y_train, X_val, y_val, n_trials)
