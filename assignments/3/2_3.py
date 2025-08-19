import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))

from MLP import MLPClassifier
import wandb

# Load the dataset
df = pd.read_csv('../../data/external/WineQT.csv')

# Handle missing values
imputer = SimpleImputer(strategy='mean')
df.iloc[:, :] = imputer.fit_transform(df)

# Define features and target
X = df.drop('quality', axis=1)  # Replace 'quality' with your target column if different
y = df['quality']

# Normalize the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encoding the target variable
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.values.reshape(-1, 1))

# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
input_size = X_train.shape[1] 
print(input_size)
model = MLPClassifier(3,[11, 64, 32, 6],0.1,'relu','sgd',32,100,input_size)
model.fit(X_train, y_train, X_val, y_val)

# Evaluate the model
train_loss, train_acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_val, y_val)

# Function to run hyperparameter tuning
def run_experiment(learning_rate, neurons_per_layer, activation_function, optimizer, epochs):
    # Initialize W&B run with current set of hyperparameters
    wandb.init(project="wineqt-mlp", config={
        "learning_rate": learning_rate,
        "neurons_per_layer": neurons_per_layer,
        "activation_function": activation_function,
        "optimizer": optimizer,
        "epochs": epochs,
    })
    
        # Initialize the model
    input_size = X_train.shape[1]  # Number of features
    output_size = y_train.shape[1]  # Number of classes in the one-hot encoded y

    mlp = MLPClassifier(
        layers=len(neurons_per_layer) - 1,
        neurons=neurons_per_layer,  # Should have the output_size as last layer neurons
        learning_rate=learning_rate,
        activation=activation_function,
        optimizer=optimizer,
        epochs=epochs,
        input_size=input_size  # Number of features
    )

    
    # Train the model
    mlp.fit(X_train, y_train, X_val, y_val)
    
    # Evaluate the model
    train_loss, train_acc = mlp.evaluate(X_train, y_train)
    val_loss, val_acc = mlp.evaluate(X_val, y_val)

    # Log final metrics
    wandb.log({
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc
    })
    
    wandb.finish()

# Example hyperparameter space for grid search
learning_rates = [0.001, 0.01, 0.1]
neurons_per_layer_configs = [[11, 64, 32, 6], [11, 128, 64, 6]]
activation_functions = ['relu', 'sigmoid', 'tanh']
optimizers = ['sgd', 'adam']
epochs_list = [50, 100]

# Grid search over hyperparameters
# for lr in learning_rates:
#     for neurons_per_layer in neurons_per_layer_configs:
#         for activation in activation_functions:
#             for optimizer in optimizers:
#                 for epochs in epochs_list:
#                     try:
#                         print(f"Running experiment with lr={lr}, neurons={neurons_per_layer}, "
#                               f"activation={activation}, optimizer={optimizer}, epochs={epochs}")
#                         run_experiment(lr, neurons_per_layer, activation, optimizer, epochs)
#                     except Exception as e:
#                         print(f"Error during experiment with lr={lr}, neurons={neurons_per_layer}, "
#                               f"activation={activation}, optimizer={optimizer}, epochs={epochs}: {e}")
