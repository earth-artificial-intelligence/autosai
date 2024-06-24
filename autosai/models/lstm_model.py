import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from .base_model import BaseModel

class LSTMModel(BaseModel):
    def __init__(self, trial, input_shape):
        super().__init__(trial, input_shape)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(units=self.trial.suggest_int('lstm_units', 16, 64, log=True),
                       input_shape=(self.input_shape, 1)))
        model.add(Dense(1, activation='linear'))

        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model
