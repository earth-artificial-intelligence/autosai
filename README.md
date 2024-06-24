# autosai
AutoML for scientific research tasks

`autosai` is a Python library that simplifies the process of AutoML model selection and hyperparameter tuning using Bayesian optimization. It supports various deep learning architectures including Dense, CNN, LSTM, Transformer, and TabNet.

## Installation

## Installation

To install `autosai`, you can use [Poetry](https://python-poetry.org/), a dependency management and packaging tool for Python projects.

```
poetry add autosai
```

This will add autosai to your Poetry project.

### Running Tests

```
poetry run pytest
```

## Usage

### Prepare Data

```
from autosai import DataPreprocessor

data_path = 'your_dataset.csv'
target_column = 'target_column'
X_train, X_val, y_train, y_val = DataPreprocessor.prepare_data(data_path, target_column)
```

### Optimize Hyperparameters

```
from autosai import ModelSelector

model_selector = ModelSelector()
best_params = model_selector.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)
print(f"Best hyperparameters: {best_params}")
```

### Create and Train Model

```
best_model_type = best_params.pop('model_type')
best_model = model_selector.create_model(best_model_type, best_params, X_train.shape[1])
best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)
```

### Save Model

```
from autosai import ModelSaver

ModelSaver.save_model(best_model, best_model_type, 'best_model')
```

### Example

Refer to the examples/example_usage.py file for a complete example.

```

### setup.py
```python
from setuptools import setup, find_packages

setup(
    name='autosai',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'tensorflow',
        'optuna',
        'pytorch-tabnet',
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A library for AutoML model selection and hyperparameter tuning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/autosai',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

```

This version of the autosai library uses classes to organize the code in a more extensible and maintainable manner. Each model type is encapsulated in its own class, and the optimization process is handled by the BayesianOptimization class. The data preprocessing and model saving functionalities are also encapsulated in their respective classes.
