from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os,sys
sys.path.append(os.path.abspath('../../models/MLP'))

from MLPmultilabel import MLP_multilabel
# Load the dataset
df = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-1/dataset.csv')
df = df.dropna()

# Define target and drop unnecessary columns
target_column = 'track_genre' 
X = df.drop(columns=[target_column, 'loudness', 'Unnamed: 0'])

# Label encoding for target variable
y, unique_labels = pd.factorize(df[target_column])

# Select only numerical columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
X_numerical = X[numerical_cols].reset_index(drop=True)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_numerical)

# Ensure preprocessed data is in a DataFrame format
X_processed = pd.DataFrame(X_scaled, columns=numerical_cols)

print(X_processed.head())



# Split the data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape}")
print(f"Validation set size: {X_val.shape}")

output_size = len(unique_labels)
# One-hot encoding y_train and y_val
y_train_encoded = np.eye(output_size)[y_train]
y_val_encoded = np.eye(output_size)[y_val]


# Assuming X_train, X_val, y_train, y_val are prepared

# Initialize MLP with input size as number of features (from X_train), 
# hidden layers of your choice, and output size equal to the number of genres
mlp_model = MLP_multilabel(input_size=X_train.shape[1], hidden_layers=[64, 32], output_size = len(unique_labels), 
                           learning_rate=0.01, activation='relu', optimizer='sgd', batch_size=32, epochs=100)

# Train the model
# mlp_model.fit(X_train, y_train, X_val, y_val)
mlp_model.fit(X_train, y_train_encoded, X_val, y_val_encoded)

# Evaluate on validation set
y_val_pred = mlp_model.predict(X_val)
accuracy, precision, recall, f1, hamming = mlp_model.evaluate_model(y_val, y_val_pred)

# Compare with previous KNN results
print(f"Validation F1 Score: {f1}")
print(f"Validation Accuracy: {accuracy}")
print(f"Validation Precision: {precision}")
print(f"Validation Recall: {recall}")

