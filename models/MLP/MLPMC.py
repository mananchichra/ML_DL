import numpy as np

class MultiLabelMLP:
    def __init__(self, layer_sizes, activations, learning_rate=0.01, epochs=100, batch_size=32):
        self.layer_sizes = layer_sizes  # [input_size, hidden1, hidden2, ..., output_size]
        self.activations = activations
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def tanh(self, z):
        return np.tanh(z)

    def tanh_derivative(self, z):
        return 1 - np.tanh(z) ** 2

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def binary_cross_entropy_loss(self, y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss

    def forward(self, X):
        self.a_values = [X]  # Store activations for each layer
        self.z_values = []   # Store pre-activations (linear combinations) for each layer
        
        for i in range(len(self.weights)):
            z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]  # Linear transformation
            self.z_values.append(z)
            
            # Apply the activation function for this layer
            if self.activations[i] == 'relu':
                a = self.relu(z)
            elif self.activations[i] == 'tanh':
                a = self.tanh(z)
            elif self.activations[i] == 'sigmoid':
                a = self.sigmoid(z)
                
            self.a_values.append(a)
        
        return self.a_values[-1]  # Output layer activation

    def backward(self, X, y, output):
        d_weights = [None] * len(self.weights)
        d_biases = [None] * len(self.biases)
        
        # Loss gradient for the output layer (assuming binary cross-entropy loss)
        delta = output - y  # For multi-label, this is just the raw error

        # Backpropagation loop
        for i in reversed(range(len(self.weights))):
            # Gradient of weights and biases
            d_weights[i] = np.dot(self.a_values[i].T, delta) / X.shape[0]
            d_biases[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
            
            # Propagate the error backward
            if i > 0:
                if self.activations[i-1] == 'relu':
                    delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
                elif self.activations[i-1] == 'tanh':
                    delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(self.z_values[i-1])
                elif self.activations[i-1] == 'sigmoid':
                    delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])

            # Update the weights and biases
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # Shuffle the data
            permutation = np.random.permutation(X.shape[0])
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            # Batch training
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # Forward pass
                output = self.forward(X_batch)
                
                # Backward pass and update weights
                self.backward(X_batch, y_batch, output)
                
            # Calculate and print the loss per epoch
            loss = self.binary_cross_entropy_loss(y, self.forward(X))  # Using binary cross-entropy loss
            print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')

    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)  # Threshold for multi-label classification

    def accuracy_score_multilabel(self, y_true, y_pred):
        return np.mean(np.all(y_true == y_pred, axis=1))  # Exact match ratio

# import numpy as np

# class MultiLabelMLP:
#     def __init__(self, layer_sizes, activations, learning_rate=0.01, epochs=100, batch_size=32):
#         self.layer_sizes = layer_sizes  # [input_size, hidden1, hidden2, ..., output_size]
#         self.activations = activations
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.weights = []
#         self.biases = []
        
#         # Initialize weights and biases
#         for i in range(len(layer_sizes) - 1):
#             self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
#             self.biases.append(np.zeros((1, layer_sizes[i + 1])))

#     def relu(self, z):
#         return np.maximum(0, z)

#     def relu_derivative(self, z):
#         return np.where(z > 0, 1, 0)

#     def tanh(self, z):
#         return np.tanh(z)

#     def tanh_derivative(self, z):
#         return 1 - np.tanh(z) ** 2

#     def sigmoid(self, z):
#         return 1 / (1 + np.exp(-z))

#     def sigmoid_derivative(self, z):
#         s = self.sigmoid(z)
#         return s * (1 - s)

#     def forward(self, X):
#         self.a_values = [X]  # Store activations for each layer
#         self.z_values = []   # Store pre-activations (linear combinations) for each layer
        
#         for i in range(len(self.weights)):
#             z = np.dot(self.a_values[-1], self.weights[i]) + self.biases[i]  # Linear transformation
#             self.z_values.append(z)
            
#             # Apply the activation function for this layer
#             if self.activations[i] == 'relu':
#                 a = self.relu(z)
#             elif self.activations[i] == 'tanh':
#                 a = self.tanh(z)
#             elif self.activations[i] == 'sigmoid':
#                 a = self.sigmoid(z)
                
#             self.a_values.append(a)
        
#         return self.a_values[-1]  # Output layer activation

#     def backward(self, X, y, output):
#         d_weights = [None] * len(self.weights)
#         d_biases = [None] * len(self.biases)
        
#         # Loss gradient for the output layer (assuming binary cross-entropy loss)
#         delta = output - y
        
#         # Backpropagation loop
#         for i in reversed(range(len(self.weights))):
#             # Gradient of weights and biases
#             d_weights[i] = np.dot(self.a_values[i].T, delta) / X.shape[0]
#             d_biases[i] = np.sum(delta, axis=0, keepdims=True) / X.shape[0]
            
#             # Propagate the error backward
#             if i > 0:
#                 if self.activations[i-1] == 'relu':
#                     delta = np.dot(delta, self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
#                 elif self.activations[i-1] == 'tanh':
#                     delta = np.dot(delta, self.weights[i].T) * self.tanh_derivative(self.z_values[i-1])
#                 elif self.activations[i-1] == 'sigmoid':
#                     delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.z_values[i-1])

#             # Update the weights and biases
#             self.weights[i] -= self.learning_rate * d_weights[i]
#             self.biases[i] -= self.learning_rate * d_biases[i]

#     def fit(self, X, y):
#         for epoch in range(self.epochs):
#             # Shuffle the data
#             permutation = np.random.permutation(X.shape[0])
#             X_shuffled = X[permutation]
#             y_shuffled = y[permutation]
            
#             # Batch training
#             for i in range(0, X.shape[0], self.batch_size):
#                 X_batch = X_shuffled[i:i+self.batch_size]
#                 y_batch = y_shuffled[i:i+self.batch_size]
                
#                 # Forward pass
#                 output = self.forward(X_batch)
                
#                 # Backward pass and update weights
#                 self.backward(X_batch, y_batch, output)
                
#             # Optionally print the loss or metrics per epoch (optional)
#             loss = np.mean(np.square(y - self.forward(X)))  # Mean squared error loss
#             print(f'Epoch {epoch + 1}/{self.epochs}, Loss: {loss}')

#     def predict(self, X):
#         output = self.forward(X)
#         return (output > 0.5).astype(int)  # Threshold for binary classification
