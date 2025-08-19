import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import wandb

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test = encoder.transform(y_test.values.reshape(-1, 1))


# Experiment 1: Effect of Activation Functions
activations = ['relu', 'sigmoid', 'tanh']

for activation in activations:
    # Initialize the MLP model with the given activation function
    mlp = MLPClassifier(layers=[12, 64, 32, 6], learning_rate=0.001, epochs=100, batch_size=32, activation=activation, optimizer='sgd')
    
    # Train the model and log performance
    mlp.fit(X_train, y_train, X_test, y_test)
    
    # Plot the loss over epochs (you'll need to save the losses within the fit method)
    plt.plot(mlp.epoch_loss, label=f'{activation} activation')

# Show the final plot for comparison
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Effect of Activation Functions on Loss')
plt.show()


# Experiment 2: Effect of Learning Rate
learning_rates = [0.1, 0.01, 0.001, 0.0001]

for lr in learning_rates:
    # Initialize the MLP model with the given learning rate
    mlp = MLPClassifier(layers=[12, 64, 32, 6], learning_rate=lr, epochs=100, batch_size=32, activation='relu', optimizer='sgd')
    
    # Train the model and log performance
    mlp.fit(X_train, y_train, X_test, y_test)
    
    # Plot the loss over epochs
    plt.plot(mlp.epoch_loss, label=f'Learning Rate = {lr}')

# Show the final plot for comparison
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Effect of Learning Rate on Loss')
plt.show()


# Experiment 3: Effect of Batch Size
batch_sizes = [16, 32, 64, 128]

for batch_size in batch_sizes:
    # Initialize the MLP model with the given batch size
    mlp = MLPClassifier(layers=[12, 64, 32, 6], learning_rate=0.001, epochs=100, batch_size=batch_size, activation='relu', optimizer='sgd')
    
    # Train the model and log performance
    mlp.fit(X_train, y_train, X_test, y_test)
    
    # Plot the loss over epochs
    plt.plot(mlp.epoch_loss, label=f'Batch Size = {batch_size}')

# Show the final plot for comparison
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Effect of Batch Size on Loss')
plt.show()
