#!pip install hmmlearn
import numpy as np
import librosa
import os
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns

import os

def access_files(directory, digit):
  file_paths = []
  for root, _, files in os.walk(directory):
    print(directory)
    for file in files[0:10]:
      if file.startswith(chr(digit+48)):
        file_paths.append(os.path.join(root, file))
  return file_paths


directory_path = '/data/external/FreeSpokenDigit/recordings' 

DIGITS = [str(i) for i in range(10)]

# Step 1: Feature Extraction (MFCC)
def extract_mfcc_features(file_path):
    audio, sr = librosa.load(file_path, sr=None)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13,n_fft=512, hop_length=256)
    return mfcc_features.T  # Transpose to make time axis the first dimension


mfcc_data = {digit: [] for digit in DIGITS}
for digit in DIGITS:
    files = access_files(directory_path, digit)
    if not files:
        print(f"Warning: No files found starting with digit {digit} in '{directory_path}'")
        continue 

    for file_path in files:
        mfcc_data[digit].append(extract_mfcc_features(file_path))
        
        
from sklearn.model_selection import train_test_split

# Assuming mfcc_data is a dictionary where each digit has a list of MFCC features
train_data = {digit: [] for digit in DIGITS}
test_data = {digit: [] for digit in DIGITS}

# Split each digit's data into training and test sets
for digit in DIGITS:
    digit_features = mfcc_data[digit]
    
    # Use train_test_split to divide each digit's data
    train_features, test_features = train_test_split(digit_features, test_size=0.2, random_state=42)
    
    train_data[digit] = train_features
    test_data[digit] = test_features

# Check the size of the training and test data to verify the split
for digit in DIGITS:
    print(f"Digit {digit}: Train samples = {len(train_data[digit])}, Test samples = {len(test_data[digit])}")
    
    
# Step 2: Visualize MFCCs (optional)
def plot_mfcc(mfcc_features, title="MFCC"):
    plt.figure(figsize=(10, 4))
    sns.heatmap(mfcc_features.T, cmap='viridis')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("MFCC Coefficients")
    plt.show()

# Visualize one example from each digit
for digit, features in mfcc_data.items():
    plot_mfcc(features[0], title=f"MFCC for Digit {digit}")
    
    
# Step 3: Train HMM Models for Each Digit
models = {}
for digit in DIGITS:
    X = np.concatenate(train_data[digit])  # Combine all samples for this digit
    lengths = [len(f) for f in train_data[digit]]  # Lengths of individual sequences
    
    # Initialize and train HMM model
    model = hmm.GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)
    model.fit(X, lengths)
    models[digit] = model
    
#  Step 4: Prediction Function

def predict_digit(mfcc_features):
    max_score = float('-inf')
    predicted_digit = None
    for digit, model in models.items():
        score = model.score(mfcc_features)
        if score > max_score:
            max_score = score
            predicted_digit = digit
    return predicted_digit

# Step 5: Evaluate Model on Test Set
# Assuming test_data is a dictionary similar to mfcc_data for test samples
correct = 0
total = 0
for digit in DIGITS:
    for mfcc_features in test_data[digit]:
        predicted_digit = predict_digit(mfcc_features)
        if predicted_digit == digit:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Recognition Accuracy: {accuracy * 100:.2f}%")




directory_path = '/data/interim/5/fol' 
DIGITS = [str(i) for i in range(7)]

# Step 1: Feature Extraction (MFCC)
def extract_mfcc_features(file_path):
    audio, sr = librosa.load(file_path, sr=10000)
    mfcc_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13,n_fft=512, hop_length=256,fmax = sr//2)
    return mfcc_features.T  # Transpose to make time axis the first dimension


mfcc_data = {digit: [] for digit in DIGITS}
for digit in DIGITS:
    files = access_files(directory_path, digit)
    if not files:
        print(f"Warning: No files found starting with digit {digit} in '{directory_path}'")
        continue 

    for file_path in files:
        mfcc_data[digit].append(extract_mfcc_features(file_path))

correct = 0
total = 0
for digit in DIGITS:
    for mfcc_features in mfcc_data[digit]:
        predicted_digit = predict_digit(mfcc_features)
        if predicted_digit == digit:
            correct += 1
        total += 1

accuracy = correct / total
print(f"Recognition Accuracy: {accuracy * 100:.2f}%")