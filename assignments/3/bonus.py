
import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))
# Import the generalized MLP class
from MLP_combined import MLP_combined  # Assuming MLP is the combined class



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load the data
df = pd.read_csv('../../data/external/WineQT.csv')

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Prepare features and target
X = data.drop(columns=['quality'])  # Assuming 'quality' is the target variable
y = data['quality']  # This should be your classification target

# One-Hot Encoding for classification
encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.values.reshape(-1, 1))  # One-hot encode the target variable

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

print(y_test.shape)
# Initialize the MLP Classifier
mlp = MLP_combined(layers=[12, 128, 32, 6], 
                   learning_rate=0.1, 
                   epochs=1000, 
                   activation='relu', 
                   output_activation='softmax')  # Use softmax for multi-class classification


# Fit the model
mlp.fit(X_train, y_train)

# Predict on validation set
y_val_pred = mlp.predict(X_val)
# Make predictions on validation set
y_val_pred = mlp.predict(X_val)

# Debugging: Check shape of predictions
print("Shape of y_val_pred:", y_val_pred.shape)

# Convert predictions from probabilities to class labels
if y_val_pred.ndim == 2:  # Ensure it's a 2D array
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)  # Get predicted class labels
else:
    print("Warning: y_val_pred is not 2D. Shape:", y_val_pred.shape)
    y_val_pred_classes = np.zeros(X_val.shape[0], dtype=int)  # Default or handle appropriately

# Convert true labels back to class indices
y_val_true_classes = np.argmax(y_val, axis=1)

y_val_pred_classes = np.argmax(y_val_pred, axis=1)  # Get predicted class labels
y_val_true_classes = np.argmax(y_val, axis=1)  # Get true class labels

# Calculate evaluation metrics
val_accuracy = accuracy_score(y_val_true_classes, y_val_pred_classes)
val_precision = precision_score(y_val_true_classes, y_val_pred_classes, average='macro', zero_division=0)
val_recall = recall_score(y_val_true_classes, y_val_pred_classes, average='macro', zero_division=0)
val_f1 = f1_score(y_val_true_classes, y_val_pred_classes, average='macro')

# Print evaluation metrics
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1 Score: {val_f1:.4f}")

# Predict on the test set
y_test_pred = mlp.predict(X_test)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

# Calculate and print test metrics
test_accuracy = accuracy_score(y_test_true_classes, y_test_pred_classes)
test_precision = precision_score(y_test_true_classes, y_test_pred_classes, average='macro', zero_division=0)
test_recall = recall_score(y_test_true_classes, y_test_pred_classes, average='macro', zero_division=0)
test_f1 = f1_score(y_test_true_classes, y_test_pred_classes, average='macro')

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
