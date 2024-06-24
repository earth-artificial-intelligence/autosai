import pytest
from autosai.models.dense_model import DenseModel
from autosai.models.cnn_model import CNNModel
from autosai.models.lstm_model import LSTMModel
from autosai.models.transformer_model import TransformerModel
from autosai.models.tabnet_model import TabNetModel

def test_dense_model():
    model = DenseModel({}, 10)
    assert model.model is not None

def test_cnn_model():
    model = CNNModel({}, 10)
    assert model.model is not None

def test_lstm_model():
    model = LSTMModel({}, 10)
    assert model.model is not None

def test_transformer_model():
    model = TransformerModel({}, 10)
    assert model.model is not None

def test_tabnet_model():
    model = TabNetModel({}, 10)
    assert model.model is not None
