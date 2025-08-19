import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, hamming_loss
import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))

from MLPmultilabel import MLP_multilabel
import wandb



df = pd.read_csv("../../data/external/advertisement.csv")


df = df.dropna()

# Preprocess the data
# 1. Handle categorical columns (e.g., 'gender', 'occupation')
categorical_columns = df.select_dtypes(include=['object', 'category']).columns
print(categorical_columns)
numerical_columns = df.select_dtypes(include=['int64', 'float64','bool']).columns
print(numerical_columns)

one_hot_columns = ['gender', 'occupation', 'city', 'education', 'most bought item']
one_hot_encoder = OneHotEncoder(sparse=False)
one_hot_encoded = one_hot_encoder.fit_transform(df[categorical_columns])

# 2. Standardize numerical features (e.g., 'age', 'income', 'purchase_amount')
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[numerical_columns])

# 3. Combine the processed features
X = np.hstack((one_hot_encoded, scaled_features))

# 4. Convert 'labels' column into multi-label binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df['labels'].apply(lambda x: x.split(', ')))

# Split the data into train and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Define input and output sizes for the model
input_size = X_train.shape[1]
output_size = y_train.shape[1]

import wandb

# Define sweep configuration
sweep_config = {
    'method': 'random',  # Choose how you want to sweep: 'grid', 'random', or 'bayes'
    'metric': {
        'name': 'accuracy',  # Metric to optimize
        'goal': 'maximize'  # Whether to minimize or maximize the metric
    },
    'parameters': {
        'learning_rate': {
            'values': [0.1]
        },
        'epochs': {
            'values': [50, 100, 150]
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'activation': {
            'values': ['relu', 'tanh', 'sigmoid']
        },
        'hidden_layers': {
            'values': [[64, 32], [128, 64, 32]]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project='MLP_multilabel_classification')
X_train_small = X_train[:100]
y_train_small = y_train[:100]

# Define a function to train the model
def train():
    # Initialize a new W&B run
    wandb.init()
    
    # Access hyperparameters from the sweep config
    config = wandb.config
    
    # Create an instance of the MLP_multilabel model with the current sweep parameters
    model = MLP_multilabel(
        input_size=input_size,  # Set the correct input size based on your data
        hidden_layers=config.hidden_layers,
        output_size=output_size,  # Set the correct output size based on your data
        learning_rate=config.learning_rate,
        activation=config.activation,
        # optimizer=config.optimizer,
        batch_size=config.batch_size,
        epochs=config.epochs
    )
    
    # Fit the model with training data
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    # Evaluate the model on validation set
    y_pred = model.predict(X_val)
    accuracy, precision, recall, f1, hamming = model.evaluate_model(y_val, y_pred)
    
    # Log the final metrics to W&B
    wandb.log({
        'final_accuracy': accuracy,
        'final_precision': precision,
        'final_recall': recall,
        'final_f1': f1,
        'final_hamming_loss': hamming
    })
    

# Start the sweep
wandb.agent(sweep_id, train)





# Create an instance of your MLP_multilabel class
model = MLP_multilabel(
    input_size=input_size,
    hidden_layers=[128, 64],  # You can adjust this based on your needs
    output_size=output_size,
    learning_rate=0.01,
    activation='relu',
    optimizer='sgd',
    batch_size=32,
    epochs=100
)

# Train the model
model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

# Evaluate on validation set
y_pred = model.predict(X_test)
model.evaluate_model(y_test, y_pred)
