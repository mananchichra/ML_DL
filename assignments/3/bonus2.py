import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.datasets import load_boston

import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))
# Import the generalized MLP class
from MLP_combined import MLP_combined  # Assuming MLP is the combined class


# Load the dataset
boston = load_boston()
X, y = boston.data, boston.target

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Initialize MLP as a regressor
mlp_regressor = MLP_combined(
    layers=[X_train.shape[1], 128, 64, 1],  # Output layer has one neuron for regression
    learning_rate=0.01,
    epochs=1000,
    activation='relu',
    output_activation='linear'  # Linear activation for regression
)

# Fit the model
mlp_regressor.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = mlp_regressor.predict(X_val)

# Calculate metrics
mse = mean_squared_error(y_val, y_val_pred)
mae = mean_absolute_error(y_val, y_val_pred)
print(f"Mean Squared Error on validation set: {mse}")
print(f"Mean Absolute Error on validation set: {mae}")

# You can also make predictions on the test set if desired
y_test_pred = mlp_regressor.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
print(f"Mean Squared Error on test set: {test_mse}")
print(f"Mean Absolute Error on test set: {test_mae}")
