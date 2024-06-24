import pytest
from autosai.model_selection import ModelSelector
from autosai.models import DenseModel, CNNModel, LSTMModel, TransformerModel, TabNetModel

def test_create_model():
    selector = ModelSelector()
    input_shape = 10

    # Test Dense Model
    model = selector.create_model('dense', {}, input_shape)
    assert isinstance(model, DenseModel)

    # Test CNN Model
    model = selector.create_model('cnn', {}, input_shape)
    assert isinstance(model, CNNModel)

    # Test LSTM Model
    model = selector.create_model('lstm', {}, input_shape)
    assert isinstance(model, LSTMModel)

    # Test Transformer Model
    model = selector.create_model('transformer', {}, input_shape)
    assert isinstance(model, TransformerModel)

    # Test TabNet Model
    model = selector.create_model('tabnet', {}, input_shape)
    assert isinstance(model, TabNetModel)

    # Test invalid model type
    with pytest.raises(ValueError):
        selector.create_model('invalid_model_type', {}, input_shape)
