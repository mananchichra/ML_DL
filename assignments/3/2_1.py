import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import sys
sys.path.append(os.path.abspath('../../models/MLP'))

from MLP import MLPClassifier

df = pd.read_csv("../../data/external/WineQT.csv")

columns_to_plot = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 
                   'pH', 'sulphates', 'alcohol', 'quality']

# Set up the figure for multiple subplots (3x3 grid)
plt.figure(figsize=(18, 12))

# Iterate over the columns and plot each as a histogram
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)  # 3 rows, 3 columns
    plt.hist(df[column], bins=20, color='skyblue', edgecolor='black')
    plt.title(f'{column}')
    # plt.xlabel(column)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Describe the dataset (this will return the full statistics)
description = df.describe()

# Display the 'mean', 'std', 'min', and 'max' from the summary table
summary_stats = description.loc[['mean', 'std', 'min', 'max']]
print(summary_stats)

# Assuming 'quality' is the label column
label_counts = df['quality'].value_counts()

# Plot the distribution of labels
plt.figure(figsize=(8, 6))
plt.bar(label_counts.index, label_counts.values, color='skyblue')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.title('Distribution of Wine Quality Labels')
plt.xticks(label_counts.index)
plt.show()



# Handling missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Separating features and target variable (assuming 'quality' is the target column)
X = df_imputed.drop(columns=['quality'])  # Adjust if 'quality' is not the target
y = df_imputed['quality']

# Standardization (mean 0, std 1)
scaler_standard = StandardScaler()
X_standardized = pd.DataFrame(scaler_standard.fit_transform(X), columns=X.columns)

# Normalization (0 to 1 range)
scaler_minmax = MinMaxScaler()
X_normalized = pd.DataFrame(scaler_minmax.fit_transform(X), columns=X.columns)

# Display the first few rows of normalized and standardized data
print("Standardized Data:")
print(X_standardized.head())

print("\nNormalized Data:")
print(X_normalized.head())
