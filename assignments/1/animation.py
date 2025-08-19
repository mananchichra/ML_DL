import imageio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import sys
import os
import time
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/linear-regression')
from linearRegression import LinearRegression

start_time = time.time()

# Function to calculate variance
def calculate_variance(errors):
    return np.var(errors)

# Function to calculate standard deviation
def calculate_std_dev(errors):
    return np.std(errors)

# Load and prepare data
data = pd.read_csv('../../data/external/regularisation.csv')
X = data.iloc[:, 0].values.reshape(-1, 1)
y = data.iloc[:, 1].values

# Shuffle and split the data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define the range of degrees and random seeds
degrees = []
for degree in range(2,21):
    degrees.append(degree)
random_seeds = [42,100,200]

# Create directories to save the images
output_dir_curve = '3_2_1_curve'
output_dir_metrics = '3_2_1_metrics'

if not os.path.exists(output_dir_curve):
    os.makedirs(output_dir_curve)

if not os.path.exists(output_dir_metrics):
    os.makedirs(output_dir_metrics)

frames_curve = []
frames_metrics = []

# Initialize lists to store metrics
mse_train_list = []
mse_test_list = []
var_train_list = []
var_test_list = []
std_dev_train_list = []
std_dev_test_list = []
degree_list = []
seed_list = []

# Iterate over seeds and degrees
for seed in random_seeds:
    np.random.seed(seed)
    for degree in degrees:
        # Initialize and fit the model
        model = LinearRegression(degree=degree, learning_rate=0.1, num_iterations=1000)
        model.fit(X_train, y_train)
        
        # Predict and calculate metrics
        predictions_train = model.predict(X_train)
        mse_train = np.mean((predictions_train - y_train) ** 2)
        var_train = calculate_variance(predictions_train - y_train)
        std_dev_train = calculate_std_dev(predictions_train - y_train)
        
        predictions_test = model.predict(X_test)
        mse_test = np.mean((predictions_test - y_test) ** 2)
        var_test = calculate_variance(predictions_test - y_test)
        std_dev_test = calculate_std_dev(predictions_test - y_test)
        
        # Append metrics to lists
        mse_train_list.append(mse_train)
        mse_test_list.append(mse_test)
        var_train_list.append(var_train)
        var_test_list.append(var_test)
        std_dev_train_list.append(std_dev_train)
        std_dev_test_list.append(std_dev_test)
        degree_list.append(degree)
        seed_list.append(seed)
        
        # Sort the training data for a smooth plot
        sorted_indices = np.argsort(X_train.flatten())
        X_train_sorted = X_train[sorted_indices]
        predictions_train_sorted = predictions_train[sorted_indices]

        # Plot the curve fitting
        plt.figure(figsize=(8, 6))
        plt.scatter(X_train, y_train, color='blue', label='Train Data')
        plt.scatter(X_test, y_test, color='green', label='Test Data')
        plt.plot(X_train_sorted, predictions_train_sorted, color='red', label=f'Fit (Degree {degree})')
        plt.title(f'Line Fitting to Curve (Degree {degree}, Seed {seed})')
        plt.legend()
        plt.grid(True)

        frame_path_curve = os.path.join(output_dir_curve, f'curve_degree_{degree}_seed_{seed}.png')
        plt.savefig(frame_path_curve)
        plt.close()
        frames_curve.append(frame_path_curve)

# Create line plots for metrics and save as images
metrics = {'MSE': (mse_train_list, mse_test_list), 
           'Variance': (var_train_list, var_test_list), 
           'Standard Deviation': (std_dev_train_list, std_dev_test_list)}

for metric_name, (train_metric, test_metric) in metrics.items():
    plt.figure(figsize=(12, 8))
    
    for i, seed in enumerate(random_seeds):
        plt.plot(degrees, train_metric[i*len(degrees):(i+1)*len(degrees)], label=f'Train Seed {seed}')
        plt.plot(degrees, test_metric[i*len(degrees):(i+1)*len(degrees)], label=f'Test Seed {seed}', linestyle='--')

    plt.xlabel('Degree')
    plt.ylabel(metric_name)
    plt.title(f'{metric_name} vs Degree')
    plt.legend()
    plt.grid(True)
    
    frame_path_metrics = os.path.join(output_dir_metrics, f'{metric_name.lower().replace(" ", "_")}.png')
    plt.savefig(frame_path_metrics)
    plt.close()
    frames_metrics.append(frame_path_metrics)

# Create the GIFs
images_curve = [imageio.imread(frame) for frame in frames_curve]
imageio.mimsave('curve_fitting_3_2_1.gif', images_curve, fps=2)

images_metrics = [imageio.imread(frame) for frame in frames_metrics]
imageio.mimsave('metrics_3_2_1.gif', images_metrics, fps=2)

print("GIFs created successfully!")

end_time = time.time()

print("Execution time:", end_time - start_time)
