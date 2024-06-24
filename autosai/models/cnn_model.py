import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from .base_model import BaseModel

class CNNModel(BaseModel):
    def __init__(self, trial, input_shape):
        super().__init__(trial, input_shape)
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv1D(filters=self.trial.suggest_int('filters', 16, 64, log=True),
                         kernel_size=self.trial.suggest_int('kernel_size', 3, 5),
                         activation='relu',
                         input_shape=(self.input_shape, 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(1, activation='linear'))

        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model
