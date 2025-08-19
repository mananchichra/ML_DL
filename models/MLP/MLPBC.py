import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wandb  # W&B integration
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MLPBinClassifier:
    def __init__(self, layers, learning_rate=0.01, epochs=1000, batch_size=32, activation='relu', optimizer='sgd', loss_function='bce'):
        self.layers = layers  # List of neuron counts in each layer, including input and output layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.loss_function = loss_function  # Loss function: 'mse' or 'bce'
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    def _initialize_weights(self):
        # Initialize weights with small random values
        return [np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01 for i in range(1, len(self.layers))]

    def _initialize_biases(self):
        # Initialize biases with zeros
        return [np.zeros((self.layers[i], 1)) for i in range(1, len(self.layers))]

    def _activation(self, Z, activation_function, is_output=False):
        if is_output:
            # Sigmoid activation for binary classification output layer
            return 1 / (1 + np.exp(-Z))  # Sigmoid activation
        elif activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'linear':
            return Z
        else:
            raise ValueError("Unsupported activation function")

    def _activation_derivative(self, Z, activation_function):
        if activation_function == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif activation_function == 'tanh':
            return 1 - np.tanh(Z)**2
        elif activation_function == 'relu':
            return np.where(Z > 0, 1, 0)
        elif activation_function == 'linear':
            return np.ones_like(Z)
        else:
            raise ValueError("Unsupported activation function")

    def _forward(self, X):
        activations = [X.T]  # Input layer activation
        Zs = []  # Store the Z values (pre-activation)

        for i in range(len(self.weights) - 1):
            Z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            activation = self._activation(Z, self.activation)
            Zs.append(Z)
            activations.append(activation)
        
        # For output layer (sigmoid for binary classification)
        Z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        activation = self._activation(Z, self.activation, is_output=True)
        Zs.append(Z)
        activations.append(activation)

        return activations, Zs

    def _backward(self, activations, Zs, y):
        m = y.shape[0]  # Number of samples
        grads = {}
        
        # Compute the derivative for the output layer
        dZ = activations[-1] - y.T  # Gradient of output layer w.r.t. cost function
        for i in reversed(range(len(self.weights))):
            grads["dW" + str(i + 1)] = np.dot(dZ, activations[i].T) / m
            grads["db" + str(i + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
            
            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                dZ = dA * self._activation_derivative(Zs[i - 1], self.activation)

        return grads

    def _update_parameters(self, grads):
        for i in range(len(self.weights)):
            # Update weights and biases using the gradients
            self.weights[i] -= self.learning_rate * grads["dW" + str(i + 1)]
            self.biases[i] -= self.learning_rate * grads["db" + str(i + 1)]

    def _compute_loss(self, activations, y):
        if self.loss_function == 'bce':
            # Binary Cross Entropy loss
            return -np.mean(y * np.log(activations[-1].T + 1e-8) + (1 - y) * np.log(1 - activations[-1].T + 1e-8))
        elif self.loss_function == 'mse':
            # Mean Squared Error loss
            print(f"y shape:{y.shape}")
            print(f"activations[-1] shape: {activations[-1].shape}")
            return np.mean((y - activations[-1].T)**2)
        else:
            raise ValueError("Unsupported loss function")

    def fit(self, X_train, y_train, X_val, y_val):
        # Assuming wandb.init() is done outside before calling this method
        # wandb.init(project="binary_classification_mse_vs_bce")
        Losses = []
        for epoch in range(self.epochs):
            # Train step
            activations_train, Zs_train = self._forward(X_train)
            grads_train = self._backward(activations_train, Zs_train, y_train)
            self._update_parameters(grads_train)
            
            # Validation step (forward pass only)
            activations_val, _ = self._forward(X_val)            
            # Compute loss and metrics for training
            train_loss = self._compute_loss(activations_train[-1], y_train)
            train_predictions = (activations_train[-1] > 0.5).astype(int)  # Binary classification
            train_accuracy = np.mean(train_predictions.T == y_train)

            # Compute loss and metrics for validation
            val_loss = self._compute_loss(activations_val, y_val)
            val_predictions = (activations_val[-1] > 0.5).astype(int)  # Binary classification
            val_accuracy = np.mean(val_predictions.T == y_val)
            
            
            
            Losses.append(train_loss)
            # print(f"epoch:{epoch+1} , train_loss: {train_loss}")

            # Log metrics to W&B
            # wandb.log({
            #     'epoch': epoch,
            #     'train_loss': train_loss,
            #     'train_accuracy': train_accuracy,
            #     'val_loss': val_loss,
            #     'val_accuracy': val_accuracy
            # })
        
        
        # Finish the W&B run when done
        # wandb.finish()
        # if epoch == (self.epochs-1):
        squared_loss = (y_val - activations_val[-1].T)**2
        # print(squared_loss.shape)
        # plt.figure(figsize=(12, 6))
        # plt.plot(range(1, len(y_val) + 1), squared_loss, label="Loss for each data point", marker='o')
        # plt.title('Squared Loss for Each Data Point')
        # plt.xlabel('Data Point')
        # plt.ylabel('Squared Loss')
        # plt.legend()
        # plt.show()
        plt.figure(figsize=(12, 6))
        plt.plot(range(1, self.epochs + 1), Losses , label=f"Loss for {self.loss_function}", marker='o')
        plt.title('Loss per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        return Losses

    def predict(self, X):
        # Forward pass to predict
        activations, _ = self._forward(X)
        return (activations[-1] > 0.5).astype(int)  # Return binary predictions
