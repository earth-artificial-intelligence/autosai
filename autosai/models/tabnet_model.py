from pytorch_tabnet.tab_model import TabNetRegressor
from .base_model import BaseModel

class TabNetModel(BaseModel):
    def __init__(self, trial, input_shape):
        super().__init__(trial, input_shape)
        self.model = self.build_model()

    def build_model(self):
        tabnet_model = TabNetRegressor(
            n_d=self.trial.suggest_int('n_d', 8, 64),
            n_a=self.trial.suggest_int('n_a', 8, 64),
            n_steps=self.trial.suggest_int('n_steps', 3, 10),
            gamma=self.trial.suggest_float('gamma', 1.0, 2.0),
            lambda_sparse=self.trial.suggest_float('lambda_sparse', 1e-6, 1e-3)
        )
        return tabnet_model

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=['mae'],
            max_epochs=100,
            patience=10,
            batch_size=256,
            virtual_batch_size=128,
            verbose=0
        )
    
    def predict(self, X_val):
        return self.model.predict(X_val)
