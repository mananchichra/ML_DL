

import numpy as np

class LinearRegression:
    def __init__(self, degree=1, learning_rate=0.01, num_iterations=1000, regularization=0.0, reg_type='l2'):
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.regularization = regularization
        self.reg_type = reg_type
        self.coefficients = None

    def fit(self, X, y):
        X_poly = self._polynomial_features(X, self.degree)
        self.coefficients = np.zeros(X_poly.shape[1])

        for _ in range(self.num_iterations):
            predictions = X_poly.dot(self.coefficients)
            errors = predictions - y
            if self.reg_type == 'l2':
                gradient = (X_poly.T.dot(errors) + self.regularization * self.coefficients) / len(y)
            elif self.reg_type == 'l1':
                gradient = (X_poly.T.dot(errors) + self.regularization * np.sign(self.coefficients)) / len(y)
            else:
                gradient = X_poly.T.dot(errors) / len(y)  
            
            self.coefficients -= self.learning_rate * gradient

    def predict(self, X):
        X_poly = self._polynomial_features(X, self.degree)
        return X_poly.dot(self.coefficients)

    def _polynomial_features(self, X, degree):
        X_poly = np.ones((X.shape[0], 1))
        for i in range(1, degree + 1):
            X_poly = np.hstack((X_poly, X ** i))
        return X_poly

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def variance(self, y_pred):
        return np.var(y_pred)

    def std_dev(self, y_pred):
        return np.std(y_pred)

    def save_model(self, file_name):
        np.save(file_name, self.coefficients)

    def load_model(self, file_name):
        self.coefficients = np.load(file_name)


