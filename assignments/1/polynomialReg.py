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

degrees = []
for i in range (1,6):
    degrees.append(i)

sorted_indices = np.argsort(X_train.flatten())
X_train_sorted = X_train[sorted_indices]

plt.figure(figsize=(10, 8))

for degree in degrees:
    model = LinearRegression(degree=degree, learning_rate=0.1, num_iterations=1000, regularization=0,reg_type = None)
    model.fit(X_train, y_train)

    # Predict on the sorted training data for plotting
    y_train_sorted_predictions = model.predict(X_train_sorted)
    
    train_predictions = model.predict(X_train)
    mse_train = np.mean((train_predictions - y_train) ** 2)
    std_train = np.std(train_predictions - y_train)
    var_train = np.var(train_predictions - y_train)
    
    # print(f'Degree:{degree}')
    # print(f'MSE: {mse_train}')
    # print(f'Variance: {var_train}')
    # print(f'Standard Deviation: {std_train}\n')

    plt.plot(X_train_sorted, y_train_sorted_predictions, label=f'Degree = {degree}, MSE = {mse_train:.4f}, Std Dev = {std_train:.4f}, Variance = {var_train:.4f}')

plt.scatter(X_train, y_train, color='blue', label='Train Data')
plt.scatter(X_val, y_val, color='orange', label='Validation Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')

plt.legend()
plt.title('Best Fit Polynomial Curves for Different Degrees (Learning Rate = 0.1)')
plt.xlabel('X')
plt.ylabel('y')
plt.grid(True)
plt.show()
