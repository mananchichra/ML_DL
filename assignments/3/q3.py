import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys
import math
# Binary classification dataset
data = pd.read_csv('../../data/external/diabetes.csv')

sys.path.append(os.path.abspath('../../models/MLP'))

from MLPReg import MLPRegressor
from MLPBC import MLPBinClassifier
import wandb

from sklearn.datasets import load_boston

# Load the dataset
boston = load_boston()

X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = pd.DataFrame(boston.target, columns=["MEDV"])

######

df = pd.DataFrame(boston.data, columns=boston.feature_names)
# Add the target (house prices) as a column to the DataFrame

description = df.describe()

# Display the 'mean', 'std', 'min', and 'max' from the summary table
summary_stats = description.loc[['mean', 'std', 'min', 'max']]
print(summary_stats)

df['MEDV'] = boston.target

# Plot the distribution of every column
plt.figure(figsize=(15, 12))

# Iterate through all the columns to plot their distributions
for i, column in enumerate(df.columns, 1):
    plt.subplot(4, 4, i)  # Adjust the grid size based on the number of columns
    plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
#####

# Handle missing values
if X.isnull().any().any():
    print("Missing values found. Filling missing values...")
    X.fillna(X.mean(), inplace=True)

def preprocess_data(X,y):
    

    # # Handle missing values
    # if X.isnull().any().any():
    #     print("Missing values found. Filling missing values...")
    #     X.fillna(X.mean(), inplace=True)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Check if they are pandas DataFrames or Series and convert only if needed
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.to_numpy()
    if isinstance(y_train, pd.Series) or isinstance(y_train, pd.DataFrame):
        y_train = y_train.to_numpy()

    if isinstance(X_val, pd.DataFrame):
        X_val = X_val.to_numpy()
    if isinstance(y_val, pd.Series) or isinstance(y_val, pd.DataFrame):
        y_val = y_val.to_numpy()

    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.to_numpy()
    if isinstance(y_test, pd.Series) or isinstance(y_test, pd.DataFrame):
        y_test = y_test.to_numpy()

    return X_train, y_train, X_val, y_val, X_test, y_test

X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X,y)

# Create MLP instance
mlp = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=1, learning_rate=0.001, activation='relu')

# Train the model
mlp.fit(X_train, y_train,X_val,y_val,epochs=500, batch_size=32)


val_predictions = mlp.predict(X_val)
val_loss = mlp.compute_loss(val_predictions, y_val)
val_mae = mean_absolute_error(y_val, val_predictions)
val_mse = mean_squared_error(y_val, val_predictions)
val_r2 = r2_score(y_val, val_predictions)

print(f"Validation Loss: {val_loss}")
print(f"Validation MAE: {val_mae}")
print(f"Validation MSE: {val_mse}")
print(f"Validation R²: {val_r2}")

# Final evaluation on test set
test_predictions = mlp.predict(X_test)
test_loss = mlp.compute_loss(test_predictions, y_test)
test_mae = mean_absolute_error(y_test, test_predictions)
test_mse = mean_squared_error(y_test, test_predictions)
test_r2 = r2_score(y_test, test_predictions)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")
print(f"Test MSE: {test_mse}")
print(f"Test R²: {test_r2}")
# Finish W&B experiment
# wandb.finish()


# # section 3.3

# # Initialize Weights & Biases run
# wandb.init(project="mlp-regression")

# Log the parameters
# config = wandb.config
# config.learning_rate = 0.01
# config.epochs = 500
# config.batch_size = 32
# config.hidden_layers = [64, 32]
# config.activation = 'relu'

# # Create MLP instance
# mlp = MLPRegressor(input_size=X_train.shape[1], 
#                    hidden_layers=config.hidden_layers, 
#                    output_size=1, 
#                    learning_rate=config.learning_rate, 
#                    activation=config.activation)

# # Training loop
# mlp.fit(X_train, y_train,X_val,y_val,epochs=config.epochs, batch_size=config.batch_size)


sweep_config = {
    'method': 'bayes',  # Can also be 'grid' or 'random'
    'metric': {
        'name': 'Test Mse',
        'goal': 'minimize'   
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.01, 0.1]
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']
        
        },
        'hidden_layers': {
            'values': [[64, 32], [128, 64], [128, 64, 32]]
        },
        'epochs':{
            'values' : [100,500,1000]
        },
        'batch_size' : {
            'values' : [32,64]
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="Q3.3 Hyperparameter-Tuning")

def train():
    # Initialize a new W&B run
    wandb.init()
    
    # Access hyperparameters from the sweep config
    config = wandb.config
    
    # Create an instance of the MLP_multilabel model with the current sweep parameters
    mlp = MLPRegressor(input_size=X_train.shape[1], 
                   hidden_layers=config.hidden_layers, 
                   output_size=1, 
                   learning_rate=config.learning_rate, 
                   activation=config.activation)
    
    
    # Training loop
    mlp.fit(X_train, y_train,X_val,y_val,epochs=config.epochs, batch_size=config.batch_size)
    predictions = mlp.predict(X_test)
    val_loss = mlp.compute_loss(predictions, y_test)
    val_rmse = math.sqrt(val_loss)
    val_r2 = r2_score(y_test, predictions)
    wandb.log({
                'Test Mse': val_loss,
                'Test Rmse': val_rmse,
                'Test R2': val_r2
                })
    
    

# Start the sweep
wandb.agent(sweep_id, train)

# Final evaluation on the test set
test_predictions = mlp.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
test_mae = np.mean(np.abs(test_predictions - y_test))

print(f"Test MSE: {test_mse}")
print(f"Test MAE: {test_mae}")


# section 3.5

from sklearn.metrics import log_loss

print(data.describe)
# # Binary classification dataset
# data = pd.read_csv('../../diabetes.csv')
# X = data.iloc[:, :-1].to_numpy # Features
# y = data.iloc[:, -1].to_numpy  # Labels (binary)
X = data.drop('DiabetesPedigreeFunction',axis=1)
y = data['DiabetesPedigreeFunction']

# Split dataset into train, validation, and test sets (similar to above)
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X, y)
print(X_train.shape , y_train.shape)
# Initialize W&B
y_train = y_train.reshape(-1, 1)  # Reshape to (number_of_samples, 1)
y_val = y_val.reshape(-1, 1)      # Reshape to (number_of_samples, 1)
y_test = y_test.reshape(-1, 1) 
# wandb.init(project="binary_classification_mse_vs_bce")

# Initialize the model
mlp_bce = MLPBinClassifier(layers=[X_train.shape[1],64, 32, 1], learning_rate=0.01, epochs=1000, loss_function='bce')
mlp_mse = MLPBinClassifier(layers=[X_train.shape[1],64, 32, 1], learning_rate=0.01, epochs=1000, loss_function='mse')

# Train the model (using BCE)
mlp_bce.fit(X_train, y_train, X_val, y_val)

# Train the model (using MSE)
mlp_mse.fit(X_train, y_train, X_val, y_val)


wandb.finish()
# step 1
# Logistic regression model using MSE Loss
mlp_mse = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[32], output_size=1, learning_rate=0.001, activation='sigmoid',loss_function='mse')
mlp_mse.fit(X_train, y_train,X_val,y_val, epochs=500, batch_size=32)

# Logistic regression model using BCE Loss
mlp_bce = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[32], output_size=1, learning_rate=0.001, activation='sigmoid', loss_function='bce')
mlp_bce.fit(X_train, y_train,X_val,y_val, epochs=500, batch_size=32)


# # step 2

# # Assuming the training loss for both models is stored in mlp_mse.losses and mlp_bce.losses

# # Plot MSE Loss
# plt.plot(range(500), mlp_mse.losses, label='MSE Loss')
# # Plot BCE Loss
# plt.plot(range(500), mlp_bce.losses, label='BCE Loss')
# plt.title('MSE vs BCE Loss over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()


# section 3.6

mlp_mse = MLPBinClassifier(layers=[X_train.shape[1],64, 32, 1], learning_rate=0.01, epochs=1000, loss_function='mse')

test_predictions = mlp_mse.fit(X_train, y_train, X_test, y_test)

# For every data point, observe the MSE Loss
test_losses = np.abs(y_test - test_predictions)  # Per-sample loss
high_loss_points = np.where(test_losses > np.mean(test_losses) + np.std(test_losses))  # High loss points
print("Length of test_losses:", len(test_losses))
print("Length of range:", len(range(len(test_losses))))

print(f"High loss points: {high_loss_points}")
print(f"Number of high MSE loss points: {len(high_loss_points[0])}")

# Plot these datapoints to observe any patterns
plt.scatter(range(len(test_losses)), test_losses)
plt.axhline(y=np.mean(test_losses) + np.std(test_losses), color='r', linestyle='--', label='High Loss Threshold')
plt.title('MSE Loss for Each Data Point')
plt.xlabel('Data Point Index')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()
