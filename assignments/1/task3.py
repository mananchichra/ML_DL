import time
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/knn')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from knn import KNN,Metrics
from splitting import StratifiedSplitter
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-1/dataset.csv')
df = df.dropna()



correlation_matrix = df.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

target_column = 'track_genre' 
X = df.drop(columns=[target_column])
y = df[target_column]

X = X.drop(columns=['energy'])

# categorical_cols = X.select_dtypes(include=['object', 'category','bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


X_numerical = X[numerical_cols].reset_index(drop=True)
X_processed = X_numerical

# scaler = StandardScaler()

# X_scaled = scaler.fit_transform(X_processed)
mean = X_processed.mean()
std = X_processed.std()
X_scaled = (X_processed - mean) / std

X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# df_new = X_scaled.copy()
# df_new['track_genre'] = y
# splitter = StratifiedSplitter(target_col='track_genre', train_size=0.8, val_size=0.1, random_state=42)
# X_train, X_val, X_test, y_train, y_val, y_test = splitter.fit_transform(df_new,y)



X_train_small = X_train.head(50000)  # Use a small subset for testing
y_train_small = y_train.head(50000)
X_val_small = X_val.head(20)
y_val_small = y_val.head(20)

best_params_small = []
knn = KNN()

try:
    for k in range(1, 22,2):
        for metric in ['euclidean','manhattan']:
            print(f"Testing k={k}, metric={metric}")
            knn.set_params(k, metric)
            knn.fit(X_train_small.values, y_train_small.values)
            y_pred_val = knn.predict(X_val_small.values)
            print ("testing")
            accuracy = Metrics.accuracy(y_val_small.values, y_pred_val)
            best_params_small.append({'k': k, 'distance_metric': metric, 'accuracy': accuracy})
            print(f"Accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"An error occurred: {e}")

print(best_params_small)


best_params = best_params_small

best_params_sorted = sorted(best_params, key=lambda x: x['accuracy'], reverse=True)


print("Top 10 {k, distance metric} pairs by validation accuracy:")
print(best_params_sorted)

for i, params in enumerate(best_params_sorted[:10], start=1):
    print(f"Rank {i}: k={params['k']}, distance_metric={params['distance_metric']}, Accuracy={params['accuracy']:.4f}")


ks = range(1,22,2)
accuracies_euclidean = [param['accuracy'] for param in best_params if param['distance_metric'] == 'euclidean']

plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies_euclidean, marker='o', label='Euclidean')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('k vs Validation Accuracy for Euclidean Distance')
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.show()

