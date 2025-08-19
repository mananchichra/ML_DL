import numpy as np
import pandas as pd
import wandb
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import precision_score, recall_score, f1_score
class MLPClassifier:
    def __init__(self, layers, learning_rate=0.01, epochs=1000, batch_size=32, activation='relu', optimizer='sgd'):
        self.layers = layers  # List of neuron counts in each layer, including input and output layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.optimizer = optimizer
        self.weights = self._initialize_weights()
        self.biases = self._initialize_biases()

    def _initialize_weights(self):
        # Initialize weights with small random values
        return [np.random.randn(self.layers[i], self.layers[i - 1]) * 0.01 for i in range(1, len(self.layers))]

    def _initialize_biases(self):
        # Initialize biases with zeros
        return [np.zeros((self.layers[i], 1)) for i in range(1, len(self.layers))]

    # def _activation(self, Z, activation_function):
    #     # Activation functions
    #     if activation_function == 'sigmoid':
    #         return 1 / (1 + np.exp(-Z))
    #     elif activation_function == 'tanh':
    #         return np.tanh(Z)
    #     elif activation_function == 'relu':
    #         return np.maximum(0, Z)
    #     elif activation_function == 'linear':
    #         return Z
    #     else:
    #         raise ValueError("Unsupported activation function")
    def _activation(self, Z, activation_function, is_output=False):
        # if is_output:
        #     expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Stable softmax
        #     return expZ / np.sum(expZ, axis=0, keepdims=True)
        if is_output:
            # Stable softmax to avoid overflow/underflow
            Z_max = np.max(Z, axis=0, keepdims=True)
            expZ = np.exp(Z - Z_max)  # Stabilizing by subtracting max(Z)
            
            # Check for NaN or inf
            if np.any(np.isnan(expZ)) or np.any(np.isinf(expZ)):
                print("Warning: NaN or inf detected in softmax computation")
            
            # Softmax computation
            epsilon = 1e-8  # Add epsilon to avoid division by zero
            return expZ / (np.sum(expZ, axis=0, keepdims=True) + epsilon)
        
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
        # Derivative of activation functions for backpropagation
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
        
        # For output layer
        Z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        activation = self._activation(Z, self.activation, is_output=True)
        Zs.append(Z)
        activations.append(activation)

        return activations, Zs
 

    # def _backward(self, activations, Zs, y):
    #     m = y.shape[0]  # Number of samples
    #     grads = {}
        
    #     # Compute the derivative for the output layer
    #     dZ = activations[-1] - y.T  # Gradient of output layer w.r.t. cost function
        
    #     for i in reversed(range(len(self.weights))):
    #         grads["dW" + str(i + 1)] = np.dot(dZ, activations[i].T) / m
    #         grads["db" + str(i + 1)] = np.sum(dZ, axis=1, keepdims=True) / m
            
    #         if i > 0:
    #             print(f"Weights[{i}] shape: {self.weights[i].shape}")
    #             print(f"dZ shape: {dZ.shape}")
    #             dA = np.dot(self.weights[i].T, dZ)
    #             dZ = dA * self._activation_derivative(Zs[i - 1], self.activation)

    #     return grads
    def _backward(self, activations, Zs, y):
        m = y.shape[0]  # Number of samples
        grads = {}

        # Compute the derivative for the output layer
        dZ = activations[-1] - y.T  # Gradient of output layer w.r.t. cost function
        # print(f"Initial dZ shape: {dZ.shape}")
        
        for i in reversed(range(len(self.weights))):
            # print(f"Weights[{i}] shape: {self.weights[i].shape}")
            # print(f"dZ shape: {dZ.shape}")
            
            grads["dW" + str(i + 1)] = np.dot(dZ, activations[i].T) / m
            grads["db" + str(i + 1)] = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 0:
                dA = np.dot(self.weights[i].T, dZ)
                # print(f"dA shape: {dA.shape}")
                dZ = dA * self._activation_derivative(Zs[i - 1], self.activation)
                # print(f"Updated dZ shape: {dZ.shape}")

        return grads


    def _update_parameters(self, grads):
        for i in range(len(self.weights)):
            # Update weights and biases using the gradients
            self.weights[i] -= self.learning_rate * grads["dW" + str(i + 1)]
            self.biases[i] -= self.learning_rate * grads["db" + str(i + 1)]

    
        
    #     wandb.finish()
    # def fit(self, X_train, y_train, X_val, y_val):
    #     # Assuming wandb.init() is done outside before calling this method
    #     if len(y_train.shape) == 1:  # Check if y_train is not one-hot encoded
    #         encoder = OneHotEncoder(sparse=False)
    #         y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        
    #     if len(y_val.shape) == 1:  # Check if y_val is not one-hot encoded
    #         y_val = encoder.transform(y_val.reshape(-1, 1))
            
    #     for epoch in range(self.epochs):
    #         # Train step
    #         activations_train, Zs_train = self._forward(X_train)
    #         grads_train = self._backward(activations_train, Zs_train, y_train)
    #         self._update_parameters(grads_train)
            
    #         # Validation step (forward pass only)
    #         activations_val, _ = self._forward(X_val)
            
    #         # Compute loss and metrics for training
    #         train_loss = -np.mean(y_train * np.log(activations_train[-1].T + 1e-8))
    #         train_predictions = np.argmax(activations_train[-1], axis=0)
    #         train_accuracy = np.mean(np.argmax(y_train, axis=1) == train_predictions)
    #         train_precision = precision_score(np.argmax(y_train, axis=1), train_predictions, average='macro', zero_division=0)
    #         train_recall = recall_score(np.argmax(y_train, axis=1), train_predictions, average='macro',zero_division=0)
    #         train_f1 = f1_score(np.argmax(y_train, axis=1), train_predictions, average='macro')

    #         # Compute loss and metrics for validation
    #         val_loss = -np.mean(y_val * np.log(activations_val[-1].T + 1e-8))
    #         val_predictions = np.argmax(activations_val[-1], axis=0)
    #         val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
    #         val_precision = precision_score(np.argmax(y_val, axis=1), val_predictions, average='macro', zero_division=0)
    #         val_recall = recall_score(np.argmax(y_val, axis=1), val_predictions, average='macro',zero_division=0)
    #         val_f1 = f1_score(np.argmax(y_val, axis=1), val_predictions, average='macro')

    #         # Log metrics to W&B
    #         wandb.log({
    #             'epoch': epoch,
    #             'train_loss': train_loss,
    #             'train_accuracy': train_accuracy,
    #             'train_precision': train_precision,
    #             'train_recall': train_recall,
    #             'train_f1': train_f1,
    #             'val_loss': val_loss,
    #             'val_accuracy': val_accuracy,
    #             'val_precision': val_precision,
    #             'val_recall': val_recall,
    #             'val_f1': val_f1
    #         })
    #     return val_accuracy
    #     # # Finish the W&B run when done
    #     # wandb.finish()
    
    
    
    
    # def fit(self, X_train, y_train, X_val, y_val):
    #     # Assuming wandb.init() is done outside before calling this method
    #     if len(y_train.shape) == 1:  # Check if y_train is not one-hot encoded
    #         encoder = OneHotEncoder(sparse=False)
    #         y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        
    #     if len(y_val.shape) == 1:  # Check if y_val is not one-hot encoded
    #         y_val = encoder.transform(y_val.reshape(-1, 1))
        
    #     num_batches = int(np.ceil(X_train.shape[0] / self.batch_size))  # Calculate number of batches
        
    #     for epoch in range(self.epochs):
    #         for batch_idx in range(num_batches):
    #             # Generate batch data
    #             start_idx = batch_idx * self.batch_size
    #             end_idx = min(start_idx + self.batch_size, X_train.shape[0])
                
    #             X_batch = X_train[start_idx:end_idx]
    #             y_batch = y_train[start_idx:end_idx]
                
    #             # Train step
    #             activations_train, Zs_train = self._forward(X_batch)
    #             grads_train = self._backward(activations_train, Zs_train, y_batch)
    #             self._update_parameters(grads_train)
            
    #         # Validation step (forward pass only after processing all batches)
    #         activations_val, _ = self._forward(X_val)
            
    #         # Compute loss and metrics for training
    #         activations_train, _ = self._forward(X_train)  # Forward pass for all train data
    #         train_loss = -np.mean(y_train * np.log(activations_train[-1].T + 1e-8))
    #         train_predictions = np.argmax(activations_train[-1], axis=0)
    #         train_accuracy = np.mean(np.argmax(y_train, axis=1) == train_predictions)
    #         train_precision = precision_score(np.argmax(y_train, axis=1), train_predictions, average='macro', zero_division=0)
    #         train_recall = recall_score(np.argmax(y_train, axis=1), train_predictions, average='macro', zero_division=0)
    #         train_f1 = f1_score(np.argmax(y_train, axis=1), train_predictions, average='macro')

    #         # Compute loss and metrics for validation
    #         val_loss = -np.mean(y_val * np.log(activations_val[-1].T + 1e-8))
    #         val_predictions = np.argmax(activations_val[-1], axis=0)
    #         val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
    #         val_precision = precision_score(np.argmax(y_val, axis=1), val_predictions, average='macro', zero_division=0)
    #         val_recall = recall_score(np.argmax(y_val, axis=1), val_predictions, average='macro', zero_division=0)
    #         val_f1 = f1_score(np.argmax(y_val, axis=1), val_predictions, average='macro')

    #         # Log metrics to W&B
    #         wandb.log({
    #             'epoch': epoch,
    #             'train_loss': train_loss,
    #             'train_accuracy': train_accuracy,
    #             'train_precision': train_precision,
    #             'train_recall': train_recall,
    #             'train_f1': train_f1,
    #             'val_loss': val_loss,
    #             'val_accuracy': val_accuracy,
    #             'val_precision': val_precision,
    #             'val_recall': val_recall,
    #             'val_f1': val_f1
    #         })

    #     return val_accuracy

    def fit(self, X_train, y_train, X_val, y_val):
        num_samples = X_train.shape[0]
        indices = np.arange(num_samples)
        
        # Assuming wandb.init() is done outside before calling this method
        if len(y_train.shape) == 1:  # Check if y_train is not one-hot encoded
            encoder = OneHotEncoder(sparse=False)
            y_train = encoder.fit_transform(y_train.reshape(-1, 1))

        if len(y_val.shape) == 1:  # Check if y_val is not one-hot encoded
            y_val = encoder.transform(y_val.reshape(-1, 1))

        self.epoch_loss = []
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)  # Shuffle the indices for each epoch
            X_train = X_train[indices]
            y_train = y_train[indices]
            
            # Mini-batch training
            for i in range(0, num_samples, self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                y_batch = y_train[i:i + self.batch_size]
                
                # Train step for each batch
                activations_train, Zs_train = self._forward(X_batch)
                grads_train = self._backward(activations_train, Zs_train, y_batch)
                self._update_parameters(grads_train)

            # Validation step (forward pass only)
            activations_val, _ = self._forward(X_val)
            
            # Compute loss and metrics for training (on entire training set)
            activations_train, _ = self._forward(X_train)
            train_loss = -np.mean(y_train * np.log(activations_train[-1].T + 1e-8))
            train_predictions = np.argmax(activations_train[-1], axis=0)
            train_accuracy = np.mean(np.argmax(y_train, axis=1) == train_predictions)
            train_precision = precision_score(np.argmax(y_train, axis=1), train_predictions, average='macro', zero_division=0)
            train_recall = recall_score(np.argmax(y_train, axis=1), train_predictions, average='macro', zero_division=0)
            train_f1 = f1_score(np.argmax(y_train, axis=1), train_predictions, average='macro')

            # Compute loss and metrics for validation
            val_loss = -np.mean(y_val * np.log(activations_val[-1].T + 1e-8))
            val_predictions = np.argmax(activations_val[-1], axis=0)
            val_accuracy = np.mean(np.argmax(y_val, axis=1) == val_predictions)
            val_precision = precision_score(np.argmax(y_val, axis=1), val_predictions, average='macro', zero_division=0)
            val_recall = recall_score(np.argmax(y_val, axis=1), val_predictions, average='macro', zero_division=0)
            val_f1 = f1_score(np.argmax(y_val, axis=1), val_predictions, average='macro')

            self.epoch_loss.append(train_loss)

            if epoch%10 == 0:
                # Log metrics to W&B
                # wandb.log({
                #     'epoch': epoch,
                #     'train_loss': train_loss,
                #     'train_accuracy': train_accuracy,
                #     'train_precision': train_precision,
                #     'train_recall': train_recall,
                #     'train_f1': train_f1,
                #     'val_loss': val_loss,
                #     'val_accuracy': val_accuracy,
                #     'val_precision': val_precision,   
                #     'val_recall': val_recall,
                #     'val_f1': val_f1
                # })
                print(f"epoch:{epoch} & val_loss: {val_loss}")
        return val_accuracy


    def predict(self, X):
        # Forward pass to predict
        activations, _ = self._forward(X)
        return np.argmax(activations[-1], axis=0)

