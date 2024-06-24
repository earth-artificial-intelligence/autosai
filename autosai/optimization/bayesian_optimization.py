import optuna
from sklearn.metrics import mean_absolute_error
import numpy as np

class BayesianOptimization:
    def __init__(self, create_model_fn):
        self.create_model_fn = create_model_fn

    def objective(self, trial, X_train, y_train, X_val, y_val):
        # Use a small random subset of the data
        subset_idx = np.random.choice(len(X_train), size=int(0.1 * len(X_train)), replace=False)
        X_subset = X_train[subset_idx]
        y_subset = y_train[subset_idx]

        model_type = trial.suggest_categorical('model_type', ['dense', 'cnn', 'lstm', 'transformer', 'tabnet'])
        model = self.create_model_fn(model_type, trial, X_train.shape[1])

        if model_type == 'tabnet':
            model.fit(X_subset, y_subset, X_val, y_val)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
        else:
            model.fit(X_subset, y_subset, validation_data=(X_val, y_val), epochs=10, batch_size=32, verbose=0)
            loss, mae = model.evaluate(X_val, y_val, verbose=0)

        return mae

    def optimize(self, X_train, y_train, X_val, y_val, n_trials):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, X_train, y_train, X_val, y_val), n_trials=n_trials)
        return study.best_params
