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
        'pytest'
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
