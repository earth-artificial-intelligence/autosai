import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, Add, Flatten, Dense
from .base_model import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, trial, input_shape):
        super().__init__(trial, input_shape)
        self.model = self.build_model()

    def build_model(self):
        inputs = Input(shape=(self.input_shape, 1))
        attention = MultiHeadAttention(num_heads=self.trial.suggest_int('num_heads', 2, 8), 
                                       key_dim=self.trial.suggest_int('key_dim', 16, 64))(inputs, inputs)
        attention = Add()([inputs, attention])
        attention = LayerNormalization(epsilon=1e-6)(attention)
        outputs = Flatten()(attention)
        outputs = Dense(1, activation='linear')(outputs)
        
        model = Model(inputs=inputs, outputs=outputs)

        learning_rate = self.trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                      loss='mean_squared_error', metrics=['mean_absolute_error'])
        return model
