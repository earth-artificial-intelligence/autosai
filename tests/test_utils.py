import pytest
from tensorflow.keras.models import Sequential
from autosai.utils import ModelSaver

def test_save_model():
    model = Sequential()
    model_saver = ModelSaver()
    model_saver.save_model(model, 'dense', 'test_model')
