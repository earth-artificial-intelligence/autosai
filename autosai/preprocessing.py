import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

class DataPreprocessor:
    @staticmethod
    def prepare_data(data_path, target_column):
        data = pd.read_csv(data_path)
        X = data.drop(columns=[target_column])
        y = data[target_column]

        # Encode categorical features
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        return X_train, X_val, y_train.to_numpy().reshape(-1, 1), y_val.to_numpy().reshape(-1, 1)
