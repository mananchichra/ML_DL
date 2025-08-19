import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/linear-regression')

from linearRegression import LinearRegression

data = pd.read_csv('../../data/external/regularisation.csv')
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=True)

results = []

degrees = range(10, 23)
alpha_values = [10, 1]

for degree in degrees:
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', label='Training Data')
    X_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    
    for alpha in alpha_values:
        for reg_type in ['l1', 'l2']: 
            model = LinearRegression(degree=degree, learning_rate=0.01, num_iterations=1000, regularization=alpha, reg_type=reg_type)
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_val_pred = model.predict(X_val)
            y_test_pred = model.predict(X_test)
            
            mse_train = model.mse(y_train, y_train_pred)
            mse_val = model.mse(y_val, y_val_pred)
            mse_test = model.mse(y_test, y_test_pred)
            
            variance_train = model.variance(y_train_pred)
            variance_val = model.variance(y_val_pred)
            variance_test = model.variance(y_test_pred)
            
            std_dev_train = model.std_dev(y_train_pred)
            std_dev_val = model.std_dev(y_val_pred)
            std_dev_test = model.std_dev(y_test_pred)
            
            results.append({
                'Degree': degree,
                'Alpha': alpha,
                'Regularization': reg_type,
                'MSE Train': mse_train,
                'MSE Validation': mse_val,
                'MSE Test': mse_test,
                'Std Dev Train': std_dev_train,
                'Std Dev Validation': std_dev_val,
                'Std Dev Test': std_dev_test,
                'Variance Train': variance_train,
                'Variance Validation': variance_val,
                'Variance Test': variance_test
            })
            
            X_poly_plot = model._polynomial_features(X_plot, degree)
            y_plot = X_poly_plot.dot(model.coefficients)
            plt.plot(X_plot, y_plot, label=f'{reg_type.upper()} (α={alpha})')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Degree {degree} with L1 and L2 Regularization')
    plt.legend()
    plt.show()

results_df = pd.DataFrame(results)
print(results_df)

# from IPython.display import display
# display(results_df)
