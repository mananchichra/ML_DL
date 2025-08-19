import numpy as np

class MLP_combined:
    def __init__(self, layers, learning_rate=0.01, epochs=1000, batch_size=32, activation='relu', output_activation=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.activations = []
        self.z_values = []
        
        # Initialize weights and biases
        for i in range(len(layers) - 1):
            weight_matrix = np.random.randn(layers[i], layers[i + 1]) * 0.01
            bias_vector = np.zeros((1, layers[i + 1]))
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)

    def activate(self, z, activation):
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'linear':
            return z  # Linear activation
        elif activation == 'softmax':  # Add softmax for multi-class classification
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # For numerical stability
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function")

    def derivative(self, z, activation):
        if activation == 'relu':
            return np.where(z > 0, 1, 0)
        elif activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-z))
            return sig * (1 - sig)
        elif activation == 'tanh':
            return 1 - np.tanh(z) ** 2
        elif activation == 'linear':
            return 1
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, X):
        self.activations = []
        self.z_values = []
        self.activations.append(X)
        a = X
        
        # Forward through hidden layers
        for i in range(len(self.weights) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.activate(z, self.activation)
            self.activations.append(a)
        
        # Output layer
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self.activate(z, self.output_activation)
        self.activations.append(a)
        return a

    def backward(self, X, y, y_pred):
        m = y.shape[0]  # number of samples
        gradients = {}
        
        # Error at the output layer
        dz = y_pred - y  # Assuming cross-entropy loss for classification
        dW = np.dot(self.activations[-2].T, dz) / m
        db = np.sum(dz, axis=0, keepdims=True) / m
        gradients['dW' + str(len(self.weights)-1)] = dW
        gradients['db' + str(len(self.biases)-1)] = db
        
        # Backpropagate through hidden layers
        for i in range(len(self.weights)-2, -1, -1):
            dz = np.dot(dz, self.weights[i+1].T) * self.derivative(self.z_values[i], self.activation)
            dW = np.dot(self.activations[i].T, dz) / m
            db = np.sum(dz, axis=0, keepdims=True) / m
            gradients['dW' + str(i)] = dW
            gradients['db' + str(i)] = db
        
        return gradients

    def update_weights(self, gradients):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients['dW' + str(i)]
            self.biases[i] -= self.learning_rate * gradients['db' + str(i)]

    def fit(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            gradients = self.backward(X, y, y_pred)
            self.update_weights(gradients)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - y_pred))  # Mean Squared Error for regression tasks
                print(f'Epoch {epoch}, Loss: {loss}')

    # def predict(self, X):
    #     y_pred = self.forward(X)
    #     if self.output_activation == 'sigmoid':
    #         return (y_pred > 0.5).astype(int)
    #     elif self.output_activation == 'softmax':
    #         return np.argmax(y_pred, axis=1)
    #     else:
    #         return y_pred

    def predict(self, X):
        output = self.forward(X)
        return output  # This should return shape (n_samples, n_classes)
