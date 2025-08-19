import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score,recall_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import sys



sys.path.append(os.path.abspath('../../models/AutoEncoders'))

from AutoEncoders import AutoEncoder

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# def train_autoencoder_knn(X, y):
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Initialize AutoEncoder
#     autoencoder = AutoEncoder(input_size=X_train.shape[1], hidden_layers=[32, 16])  # Hidden layers can be adjusted
#     autoencoder.fit(X_train)

#     # Get the reduced dataset
#     X_train_reduced = autoencoder.transform(X_train)
#     X_test_reduced = autoencoder.transform(X_test)

#     # Train KNN on the reduced dataset
#     knn = KNeighborsRegressor(n_neighbors=5)
#     knn.fit(X_train_reduced, y_train)

#     # Make predictions
#     y_pred = knn.predict(X_test_reduced)

#     # Evaluate performance
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     print(f"KNN with AutoEncoder: MSE: {mse}, R2: {r2}")

# def train_mlp(X, y):
#     # Split the data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     # Standardize the data
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Initialize MLP
#     mlp = MLPRegressor(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size=1, learning_rate=0.001, activation='relu')
#     mlp.fit(X_train, y_train, epochs=500, batch_size=32)

#     # Predict and evaluate on test set
#     y_pred = mlp.predict(X_test)
#     test_loss = mlp.compute_loss(y_test, y_pred)
#     print(f"MLP Classifier: Test Loss: {test_loss}")

def preprocess_data(X,y):
    

    # # Handle missing values
    # if X.isnull().any().any():
    #     print("Missing values found. Filling missing values...")
    #     X.fillna(X.mean(), inplace=True)

    # Split the data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
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


df = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-1/dataset.csv')
df = df.dropna()

target_column = 'track_genre' 
X = df.drop(columns=[target_column])
X = X.drop(columns=['loudness'])
X = X.drop(columns = ['Unnamed: 0'])
print(X.columns)

#label encoding
y, unique_labels = pd.factorize(df[target_column])


categorical_cols = X.select_dtypes(include=['object', 'category','bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_numerical = X[numerical_cols].reset_index(drop=True)
X_processed = X_numerical
print(X_processed.shape)

#4.3
# Load and preprocess the data
X_train, y_train, X_val, y_val, X_test, y_test = preprocess_data(X_processed,y)  # Ensure preprocess_data is defined

# # Train the AutoEncoder + KNN
# train_autoencoder_knn(X_train, y_train)

sys.path.append(os.path.abspath('../../models/MLP'))
from MLPReg import MLPRegressor

# # Train the MLP Classifier
# train_mlp(X_train, y_train)

print(X_train.shape[1])
hidden_layers = [64,32]
# print(hidden_layers[-1])
autoencoder = AutoEncoder(input_size=X_train.shape[1], hidden_layers=hidden_layers,output_size = 8)

# Train the autoencoder
autoencoder.fit(X_train, epochs=250, batch_size=32)

# Transform the dataset using the trained encoder
reduced_X_train = autoencoder.get_latent(X_train)
reduced_X_test = autoencoder.get_latent(X_test)
reduced_X_val = autoencoder.get_latent(X_val)
# data combinin

# # Convert reduced datasets into DataFrames
# reduced_X_train_df = pd.DataFrame(reduced_X_train)
# reduced_X_test_df = pd.DataFrame(reduced_X_test)
# reduced_X_val_df = pd.DataFrame(reduced_X_val)
# # Convert y_train and y_test into DataFrames
# y_train_df = pd.DataFrame(y_train, columns=target_column)  # Name the label column as 'target'
# y_test_df = pd.DataFrame(y_test, columns=target_column)
# y_val_df = pd.DataFrame(y_val, columns=target_column)
# # Combine reduced_X_train with y_train, and reduced_X_test with y_test
# combined_train_df = pd.concat([reduced_X_train_df, y_train_df], axis=1)
# combined_test_df = pd.concat([reduced_X_test_df, y_test_df], axis=1)

# # Save the combined datasets to CSV files
# combined_train_df.to_csv('combined_train_reduced.csv', index=False)
# combined_test_df.to_csv('combined_test_reduced.csv', index=False)
# KNN Classification
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(reduced_X_train, y_train)

# Evaluate KNN on the reduced dataset
knn_predictions = knn.predict(reduced_X_test)
knn_accuracy = np.mean(knn_predictions == y_test)
print(f"KNN Accuracy on Reduced Dataset: {knn_accuracy * 100:.2f}%")
# Precision
precision = precision_score(y_test, knn_predictions, average='weighted')  # 'weighted' for multi-class
print(f"KNN Precision on Reduced Dataset: {precision * 100:.2f}%")

# Recall
recall = recall_score(y_test, knn_predictions, average='weighted')
print(f"KNN Recall on Reduced Dataset: {recall * 100:.2f}%")

# F1 Score
f1 = f1_score(y_test, knn_predictions, average='weighted')
print(f"KNN F1 Score on Reduced Dataset: {f1 * 100:.2f}%")




# 4.4

# X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.2, random_state=42)
# X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# X_val = scaler.transform(X_val)

# encoder = OneHotEncoder(sparse=False)
# y_train = encoder.fit_transform(y_train.values.reshape(-1, 1))
# y_test = encoder.transform(y_test.values.reshape(-1, 1))
# y_val = encoder.transform(y_val.values.reshape(-1, 1))

X_train, X_temp, y_train, y_temp = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print("hi")
print(X_train.shape)
print(X_temp.shape)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_val = scaler.transform(X_val)

# OneHotEncoding the labels
encoder = OneHotEncoder(sparse=False)
y_train = encoder.fit_transform(y_train.reshape(-1, 1))  # No need for .values
y_test = encoder.transform(y_test.reshape(-1, 1))
y_val = encoder.transform(y_val.reshape(-1, 1))

# y_train = y_train.reshape(-1, 1)  # Reshape to (number_of_samples, 1)
# y_val = y_val.reshape(-1, 1)      # Reshape to (number_of_samples, 1)
# y_test = y_test.reshape(-1, 1) 
sys.path.append(os.path.abspath('../../models/MLP'))
from MLP import MLPClassifier
# MLP Regression on Original Data
mlp = MLPClassifier(layers=[X_train.shape[1],64, 32,1])

print(X_val.shape)
# Train the MLP
accuracy_score = mlp.fit(X_train, y_train, X_val, y_val)

print(f"accuracy is : {accuracy_score}")

# Predict and evaluate on validation set
val_predictions = mlp.predict(X_val)
val_loss = mlp.compute_loss(val_predictions, y_val)
print(f"Validation Loss: {val_loss}")

# Final evaluation on test set
test_predictions = mlp.predict(X_test)
test_loss = mlp.compute_loss(test_predictions, y_test)
print(f"Test Loss: {test_loss}")