import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from .base_model import BaseModel

class DenseModel(BaseModel):
    def __init__(self, trial, input_shape):
        super().__init__(trial, input_shape)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        num_layers = self.trial.suggest_int('num_layers', 1, 5)
        for i in range(num_layers):
            num_units = self.trial.suggest_int(f'num_units_l{i}', 16, 128, log=True)
            if i == 0:
                model.add(Dense(num_units, activation='relu', input_shape=(self.input_shape,)))
            else:
                model.add(Dense(num_units, activation='relu'))
            dropout_rate = self.trial.suggest_float(f'dropout_rate_l{i}', 0.1, 0.5)
            model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model
