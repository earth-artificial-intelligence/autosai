from autosai import ModelSelector, DataPreprocessor, ModelSaver

# Load and prepare your data
data_path = 'your_dataset.csv'  # Replace with your actual data source
target_column = 'target_column'  # Replace with your actual target column
X_train, X_val, y_train, y_val = DataPreprocessor.prepare_data(data_path, target_column)

# Run the Bayesian optimization
model_selector = ModelSelector()
best_params = model_selector.optimize_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50)
print(f"Best hyperparameters: {best_params}")

# Train the best model on the full dataset
best_model_type = best_params.pop('model_type')
best_model = model_selector.create_model(best_model_type, best_params, X_train.shape[1])
best_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)

# Save the best model
ModelSaver.save_model(best_model, best_model_type, 'best_model')
