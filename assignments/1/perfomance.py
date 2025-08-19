import time
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/knn')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from splitting import StratifiedSplitter
# from knnPerformance import KNNPerformanceEvaluator
from knn import KNNVectorized,Metrics
from knn import KNN , KNNPerformanceEvaluator
import matplotlib.pyplot as plt

initial_knn_model = KNN()
initial_knn_model.set_params(3,'manhattan')

best_knn_model = KNN()
best_knn_model.set_params(19,'manhattan')

optimised_knn_model = KNNVectorized()
optimised_knn_model.set_params(5,'euclidean')
optimised_knn_model.set_batch_size(200)

# le = LabelEncoder()

df = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-1/dataset.csv')
df = df.dropna()

target_column = 'track_genre' 
X = df.drop(columns=[target_column])
X = X.drop(columns=['energy'])
X = X.drop(columns = ['Unnamed: 0'])

track_genre = df[target_column]

# #label encoding
# le.fit(track_genre)
# y = le.transform(track_genre)
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

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
# df_new = X_scaled.copy()
# df_new['track_genre'] = y
# splitter = StratifiedSplitter(target_col='track_genre', train_size=0.8, val_size=0.1, random_state=42)
# X_train, X_val, X_test, y_train, y_val, y_test = splitter.fit_transform(df_new,y)

evaluator = KNNPerformanceEvaluator(X_train[:10000], y_train[:10000], X_test[:100], y_test[:100])

evaluator.add_model('Initial KNN Model', initial_knn_model)
evaluator.add_model('Best KNN Model', best_knn_model)
evaluator.add_model('Most Optimized KNN Model', optimised_knn_model)
evaluator.add_model('Sklearn KNN Model', KNeighborsClassifier(n_neighbors=5))

evaluator.measure_inference_time()
evaluator.plot_inference_times()

train_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
evaluator.measure_inference_time_vs_train_size(train_sizes)
evaluator.plot_inference_time_vs_train_size(train_sizes)
