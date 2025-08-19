
import time
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/knn')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from splitting import StratifiedSplitter
from knn import KNNVectorized,Metrics
from knn import KNNPerformanceEvaluator
import matplotlib.pyplot as plt


df = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-1/dataset.csv')
df = df.dropna()

target_column = 'track_genre' 
X = df.drop(columns=[target_column])
X = X.drop(columns=['loudness'])
X = X.drop(columns = ['Unnamed: 0'])

# print(X.columns)

#label encoding
y, unique_labels = pd.factorize(df[target_column])


categorical_cols = X.select_dtypes(include=['object', 'category','bool']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


X_numerical = X[numerical_cols].reset_index(drop=True)
X_processed = X_numerical

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_processed)
mean = X_processed.mean()
std = X_processed.std()
X_scaled = (X_processed - mean) / std

X_scaled = pd.DataFrame(X_scaled, columns=X_processed.columns)

sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/pca')
from pca import PCA

pca_full = PCA(n_components=10)
pca_full.fit(X_scaled)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio) + 1), pca_full.explained_variance_ratio, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()

pca_reduced = PCA(n_components=8)
pca_reduced.fit(X_scaled)
X_scaled2 = pca_reduced.transform(X_scaled)

# X_scaled2 = X_scaled2[:10000]
# X_scaled = X_scaled[:10000]

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_train2, X_temp2, y_train2, y_temp2 = train_test_split(X_scaled2, y, test_size=0.2, random_state=42, stratify=y)
X_val2, X_test2, y_val2, y_test2 = train_test_split(X_temp2, y_temp2, test_size=0.5, random_state=42, stratify=y_temp2)
# df_new = X_scaled.copy()
# df_new['track_genre'] = df[target_column]
# splitter = StratifiedSplitter(target_col='track_genre', train_size=0.8, val_size=0.1, random_state=42)
# X_train, X_val, X_test, y_train, y_val, y_test = splitter.fit_transform(df_new,df[target_column])


# print("hello")
# X_train_small = X_train_pca_df.head(20000)  # Use a small subset for testing


best_params_small = []


inference_times = {}
k = 19
metric = 'manhattan'
knn = KNNVectorized()

print(f"Testing k={k}, metric={metric}")
start_time1 = time.time()
knn.set_params(k, metric)
knn.fit(X_train.values, y_train)
knn.set_batch_size(200)
y_pred_val = knn.predict(X_val.values)
print ("testing")
accuracy = Metrics.accuracy(y_val, y_pred_val)
micro_f1 = Metrics.f1_score(y_val,y_pred_val,'micro')
macro_f1 = Metrics.f1_score(y_val,y_pred_val,'macro')
recall = Metrics.recall(y_val,y_pred_val,'macro')
precision = Metrics.precision(y_val,y_pred_val,'macro') 
end_time1 = time.time()           
best_params_small.append({'k': k, 'distance_metric': metric, 'accuracy': accuracy})
print(f"Accuracy: {accuracy:.4f}")
print(f"Micro_f1:{micro_f1:.4f}")
print(f"Macro_f1:{macro_f1:.4f}")
print(f"recall:{recall:.4f}")
print(f"precision:{precision:.4f}")
            
print(f"Testing k={k}, metric={metric}")
start_time2 = time.time()
knn.set_params(k, metric)
knn.fit(X_train2.values, y_train2)
knn.set_batch_size(200)
y_pred_val2 = knn.predict(X_val2.values)
print ("testing")
accuracy = Metrics.accuracy(y_val2, y_pred_val2)
micro_f1 = Metrics.f1_score(y_val2,y_pred_val2,'micro')
macro_f1 = Metrics.f1_score(y_val2,y_pred_val2,'macro')
recall = Metrics.recall(y_val2,y_pred_val2,'macro')
precision = Metrics.precision(y_val2,y_pred_val2,'macro') 
end_time2 = time.time()           
best_params_small.append({'k': k, 'distance_metric': metric, 'accuracy': accuracy})
print(f"Accuracy: {accuracy:.4f}")
print(f"Micro_f1:{micro_f1:.4f}")
print(f"Macro_f1:{macro_f1:.4f}")
print(f"recall:{recall:.4f}")
print(f"precision:{precision:.4f}")

inference_times['Complete Dataset'] = end_time1-start_time1
inference_times['Reduced Dataset'] = end_time2-start_time2

plt.figure(figsize=(10, 6))
plt.bar(inference_times.keys(), inference_times.values(), color='skyblue')
plt.xlabel('Model')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time for Different KNN Models')
plt.show()