import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))

from MLP import MLPClassifier
import wandb

df = pd.read_csv('../../data/external/WineQT.csv')
# Handling missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

X = data.drop(columns=['quality'])
y = data['quality']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test = encoder.transform(y_test.values.reshape(-1, 1))
y_val = encoder.transform(y_val.values.reshape(-1, 1))


mlp = MLPClassifier(layers=[12, 128, 32, 6], 
                    learning_rate=0.1, 
                    epochs=1000, 
                    activation='relu')

# Fit model with logging
mlp.fit(X_train, y_train, X_test, y_test)
activations_val, _ = mlp._forward(X_test)
val_predictions = np.argmax(activations_val[-1], axis=0)
val_accuracy = np.mean(np.argmax(y_test, axis=1) == val_predictions)
val_precision = precision_score(np.argmax(y_test, axis=1), val_predictions, average='macro', zero_division=0)
val_recall = recall_score(np.argmax(y_test, axis=1), val_predictions, average='macro', zero_division=0)
val_f1 = f1_score(np.argmax(y_test, axis=1), val_predictions, average='macro')

print(
                f"test_accuracy: {val_accuracy}"
                f"test_precision: {val_precision}"
                f"test_recall: {val_recall}"
                f"test_f1': {val_f1}"
                )
# # Load the dataset
# data = pd.read_csv('../../data/external/WineQT.csv')

# unique_classes = data['quality'].nunique()
# print(unique_classes)  # This should print 6

# # Separate features (X) and target (y)
# X = data.drop(columns=['quality'])  # Features (including 'Id' column as you're keeping it)
# y = data['quality']  # Target (labels)

# # Normalize or Standardize features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# # One-hot encode the labels
# encoder = OneHotEncoder(sparse=False)
# y_one_hot = encoder.fit_transform(y.values.reshape(-1, 1))

# # Split the dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)



# # mlp = MLPClassifier(layers=[12, 64, 32, 10], learning_rate=0.01, epochs=1000, activation='relu')
# # mlp.fit(X_train, y_one_hot)
# # Initialize the MLP with your desired architecture
# mlp = MLPClassifier(layers=[12, 64, 32, 6], learning_rate=0.01, epochs=1000)

# # Fit the model on the training data
# mlp.fit(X_train, y_train)

# # Predict the test set
# predictions = mlp.predict(X_test)

# # Decode the predictions (convert one-hot encoded predictions back to original label form)
# # predicted_labels = encoder.inverse_transform(predictions.reshape(-1, 1)).flatten()
# # Map predictions to original labels
# predicted_labels = encoder.categories_[0][predictions]


# # Optionally, you can compare with y_test (also decode it)
# y_test_labels = encoder.inverse_transform(y_test).flatten()

# # predictions = mlp.predict(X_test)   

# accuracy = accuracy_score(y_test_labels, predicted_labels)
# print(f"Accuracy: {accuracy * 100:.2f}%")

# plt.figure(figsize=(10, 5))
# plt.plot(predicted_labels, 'bo', label='Predicted', alpha=0.6)
# plt.plot(y_test_labels, 'ro', label='Actual', alpha=0.3)
# plt.title('Predicted vs Actual Wine Quality')
# plt.xlabel('Sample Index')
# plt.ylabel('Wine Quality')
# plt.legend()
# plt.show()





# # Initialize a new W&B run
# wandb.init(project="wine-quality-mlp", config={
#     "learning_rate": 0.01,
#     "epochs": 1000,
#     "batch_size": 32,
#     "activation": "relu",
#     "optimizer": "sgd",
#     "layers": [12, 64, 32, 6]
# })
# config = wandb.config  # Access hyperparameters from the W&B config


sweep_config = {
    'method': 'random',  # 'grid' or 'bayes' can also be used
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.001, 0.01, 0.1]
        },
        'epochs': {
            'values': [100, 500, 1000]
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']
        },
        'optimizer': {
            'values': ['sgd']
        },
        'hidden_layer_1': {
            'values': [32, 64, 128]
        },
        'hidden_layer_2': {
            'values': [16, 32, 64]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="wine-quality-HyperparameterTuning")
# wandb.init()
def train():
    
    # Start a W&B run
    wandb.init()
    
    # Access hyperparameters
    config = wandb.config
    # Load data, apply preprocessing, etc.
    df = pd.read_csv('../../data/external/WineQT.csv')
    # Handling missing values by imputing with the mean
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    X = data.drop(columns=['quality'])
    y = data['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_test = encoder.transform(y_test.values.reshape(-1, 1))

    # Get hyperparameters from W&B
    config = wandb.config
    
    mlp = MLPClassifier(layers=[12, config.hidden_layer_1, config.hidden_layer_2, 6], 
                        learning_rate=config.learning_rate, 
                        epochs=config.epochs, 
                        activation=config.activation,
                        optimizer=config.optimizer)
    
    # Fit model with logging
    mlp.fit(X_train, y_train, X_test, y_test)
    activations_val, _ = mlp._forward(X_test)
    val_predictions = np.argmax(activations_val[-1], axis=0)
    val_accuracy = np.mean(np.argmax(y_test, axis=1) == val_predictions)
    val_precision = precision_score(np.argmax(y_test, axis=1), val_predictions, average='macro', zero_division=0)
    val_recall = recall_score(np.argmax(y_test, axis=1), val_predictions, average='macro', zero_division=0)
    val_f1 = f1_score(np.argmax(y_test, axis=1), val_predictions, average='macro')
    
    wandb.log({
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,   
                'val_recall': val_recall,
                'val_f1': val_f1
                })

# def sweep_train():
#     # Initialize W&B run with hyperparameters
#     wandb.init()
#     config = wandb.config
    
#     # Create the model with the current hyperparameters
#     mlp = MLPClassifier(layers=config.layers, learning_rate=config.learning_rate, 
#                         epochs=config.epochs, batch_size=config.batch_size, 
#                         activation=config.activation, optimizer=config.optimizer)
    
#     # Train the model with the training set
#     mlp.fit(X_train, y_train, X_val, y_val)

# # Start the sweep agent to train with multiple configurations
# wandb.agent(sweep_id, function=sweep_train)

wandb.agent(sweep_id, function=train, count=20)  # Will run 20 experiments


# 2.5
