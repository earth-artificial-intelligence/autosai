import pytest
from autosai.preprocessing import DataPreprocessor

def test_prepare_data():
    import pandas as pd
    from io import StringIO

    csv_data = """feature1,feature2,target
                  1,2,3
                  4,5,6
                  7,8,9"""
    data = pd.read_csv(StringIO(csv_data))
    data.to_csv('test_data.csv', index=False)

    X_train, X_val, y_train, y_val = DataPreprocessor.prepare_data('test_data.csv', 'target')
    assert X_train.shape[1] == 2
    assert y_train.shape[1] == 1
