import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb  # W&B integration
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Initialize W&B for experiment tracking
# wandb.init(project="MLP_Regression_Boston_Housing")

# Activation functions
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def tanh(z):
    return np.tanh(z)

def tanh_derivative(z):
    return 1 - np.tanh(z) ** 2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

# MLP class for regression
class MLPRegressor:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, activation='relu'):
        self.learning_rate = learning_rate
        self.activation = activation
        self.weights = []
        self.biases = []
        self.layers = [input_size] + hidden_layers + [output_size]

        # Initialize weights and biases
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i+1]) * 0.01)
            self.biases.append(np.zeros((1, self.layers[i+1])))

    def _activation(self, z):
        if self.activation == 'sigmoid':
            return sigmoid(z)
        elif self.activation == 'tanh':
            return tanh(z)
        else:  # Default is ReLU
            return relu(z)

    def _activation_derivative(self, z):
        if self.activation == 'sigmoid':
            return sigmoid_derivative(z)
        elif self.activation == 'tanh':
            return tanh_derivative(z)
        else:  # Default is ReLU
            return relu_derivative(z)

    def forward(self, X):
        self.a = [X]  # Input layer activations
        self.z = []   # Linear combinations

        # Forward propagation
        for i in range(len(self.weights)):
            z = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.z.append(z)
            a = self._activation(z) if i < len(self.weights) - 1 else z  # No activation for output layer
            self.a.append(a)

        return self.a[-1]

    def backward(self, X, y):
        m = X.shape[0]
        dz = self.a[-1] - y  # Output layer error
        dw = np.dot(self.a[-2].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        self.dw = [dw]
        self.db = [db]

        # Backpropagation through layers
        for i in range(len(self.weights) - 2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * self._activation_derivative(self.z[i])
            dw = np.dot(self.a[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            self.dw.insert(0, dw)
            self.db.insert(0, db)

    def update_params(self):
        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * self.dw[i]
            self.biases[i] -= self.learning_rate * self.db[i]

    def fit(self, X,y,X_val,y_val,epochs=100, batch_size=None):
        for epoch in range(epochs):
            if batch_size:
                for i in range(0, X.shape[0], batch_size):
                    X_batch = X[i:i+batch_size]
                    y_batch = y[i:i+batch_size]
                    self.forward(X_batch)
                    self.backward(X_batch, y_batch)
                    self.update_params()
            else:
                self.forward(X)
                self.backward(X, y)
                self.update_params()

            # Logging to Weights & Biases
            if epoch % 10 == 0:
                predictions = self.predict(X_val)
                val_loss = self.compute_loss(predictions, y_val)
                val_rmse = math.sqrt(val_loss)
                val_r2 = r2_score(y_val, predictions)
                # wandb.log({
                # 'epoch': epoch,
                # 'val_loss': val_loss,
                # 'val_rmse': val_rmse,
                # 'val_r2': val_r2
                # })
                print(f"Epoch {epoch}, Validation Loss: {val_loss}")

    def compute_loss(self, predictions, y):
        return np.mean((predictions - y) ** 2)  # Mean Squared Error (MSE)

    def compute_mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def compute_rmse(self, y_true, y_pred):
        return np.sqrt(self.compute_mse(y_true, y_pred))

    def compute_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2_score = 1 - (ss_res / ss_tot)
        return r2_score

    
    def predict(self, X):
        return self.forward(X)

# # Data Preprocessing
# def preprocess_data():
#     from sklearn.datasets import load_boston
#     boston = load_boston()    
#     X = pd.DataFrame(boston.data, columns=boston.feature_names)
#     y = pd.DataFrame(boston.target, columns=["MEDV"])

#     # Summary statistics
#     print(X.describe())

#     # Split the data
#     X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.transform(X_val)
#     X_test = scaler.transform(X_test)

#     return X_train, y_train, X_val, y_val, X_test, y_test

if __name__ == "__main__":
    # Load and preprocess the data
    X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data()

    # Create MLP instance
    mlp = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=1, learning_rate=0.001, activation='relu')

    # Train the model
    mlp.fit(X_train, y_train, epochs=500, batch_size=32)

    # Predict and evaluate on validation set
    val_predictions = mlp.predict(X_val)
    val_loss = mlp.compute_loss(val_predictions, y_val)
    print(f"Validation Loss: {val_loss}")

    # Final evaluation on test set
    test_predictions = mlp.predict(X_test)
    test_loss = mlp.compute_loss(test_predictions, y_test)
    print(f"Test Loss: {test_loss}")

    # Finish W&B experiment
    wandb.finish()
