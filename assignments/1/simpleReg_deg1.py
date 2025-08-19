import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/linear-regression')

from linearRegression import LinearRegression

data = pd.read_csv('../../data/external/linreg.csv')
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

learning_rates = [0.001, 0.005, 0.1]

sorted_indices = np.argsort(X_train.flatten())
X_train_sorted = X_train[sorted_indices]

plt.figure(figsize=(10, 8))

for lr in learning_rates:
    model = LinearRegression(degree=1, learning_rate=lr, num_iterations=1000, regularization=0,reg_type = None)
    model.fit(X_train, y_train)

    y_train_predictions = model.predict(X_train)
    y_train_sorted_predictions = model.predict(X_train_sorted)

    mse = np.mean((y_train_predictions - y_train) ** 2)
    variance = np.var(y_train_predictions - y_train)
    std_dev = np.std(y_train_predictions - y_train)

    plt.plot(X_train_sorted, y_train_sorted_predictions, label=f'LR = {lr} (MSE={mse:.4f}, Var={variance:.4f}, StdDev={std_dev:.4f})')

    print(f'Learning Rate: {lr}')
    print(f'MSE: {mse}')
    print(f'Variance: {variance}')
    print(f'Standard Deviation: {std_dev}')
    print('-' * 30)

plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_val, y_val, color='orange', label='Validation Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')

plt.legend()
plt.title('Best Fit Lines for Different Learning Rates with MSE, Variance, and Std Dev')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()
