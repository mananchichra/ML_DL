import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os
import time
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/knn')


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from splitting import StratifiedSplitter
from knn import KNNVectorized,Metrics
from knn import KNN
import matplotlib.pyplot as plt
output_dir = "figures/heatmaps"
os.makedirs(output_dir, exist_ok=True)

# le = LabelEncoder()
df = pd.read_csv("../../data/external/Spotify-1/dataset.csv") 
df = df.drop(columns = ['Unnamed: 0'])

target_variable = 'track_genre'


numerical = df.select_dtypes(include=['int64','float64']).columns
categories = df.select_dtypes(include=['category','object','bool']).columns
print(df.columns)

numerical_features = ['popularity', 'duration_ms', 'danceability', 'energy', 'key', 'loudness', 'mode', 
                       'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 
                       'time_signature']

for feature in numerical_features:
    break
    plt.figure(figsize=(10, 6))
    plt.hist(df[feature], bins=30, edgecolor='k', alpha=0.7)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


plt.figure(figsize=(12, 10))
correlation_matrix = df[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

####################

popular_songs_df = df[df['popularity'] > 80]

if popular_songs_df.empty:
    print("No songs with popularity > 80")
else:
    genre_counts = popular_songs_df[target_variable].value_counts()
    filtered_genres = genre_counts[genre_counts > 10]

genre_data = pd.DataFrame({
    'Genre': filtered_genres.index,
    'Count': filtered_genres.values
})

plt.figure(figsize=(12, 8))
sns.barplot(x='Genre', y='Count', data=genre_data, palette='viridis')

plt.xticks(rotation=45, ha='right')

plt.title('Number of Popular Songs (Popularity > 80) by Genre (More than 10 Songs)')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.grid(axis='y')

plt.show()


## explicit

explicit_df = df[df['explicit'] == True]

if explicit_df.empty:
    print("No songs with popularity > 80")
else:
    genre_counts = explicit_df[target_variable].value_counts()
    filtered_genres = genre_counts[genre_counts > 80]

genre_data = pd.DataFrame({
    'Genre': filtered_genres.index,
    'Count': filtered_genres.values
})

plt.figure(figsize=(12, 8))
sns.barplot(x='Genre', y='Count', data=genre_data, palette='viridis')

plt.xticks(rotation=45, ha='right')

plt.title('Number of Explicit Songs by Genre (More than 80 Songs)')
plt.xlabel('Genre')
plt.ylabel('Number of Songs')
plt.grid(axis='y')

plt.show()

# ##



###########################

categorical_features = ['artists', 'album_name', 'track_name', 'explicit']

# Map target variable to numerical for scatter plots
df[target_variable] = df[target_variable].astype('category')
df[target_variable] = df[target_variable].cat.codes

for feature in numerical_features:
    break
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df[feature], y=df[target_variable])
    
    plt.title(f'{feature} vs. {target_variable}')
    plt.xlabel(feature)
    plt.ylabel(target_variable)
    plt.grid(True)
    plt.show()


#outlier
for feature in numerical_features:
    break
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df[feature])
    plt.title(f'Boxplot of {feature}')
    plt.xlabel(feature)
    plt.grid(True)
    plt.show()

target_corr = df.corr()[target_variable]
print(target_corr)
ranked_features = target_corr.abs().sort_values(ascending=False)
print(ranked_features)


##############


initial_knn_model = KNN()
initial_knn_model.set_params(3,'manhattan')


#testing spotify-2 dataset

test = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-2/test.csv')
train = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-2/train.csv')
val = pd.read_csv('/home/mananchichra/Downloads/SMAI_ASSIGNMENT/data/external/Spotify-2/validate.csv')

print(test.shape)
print(val.shape)
print(train.shape)
# //113180
test = test.dropna()
train = train.dropna()
val = val.dropna()

target_column = 'track_genre' 

X_train = train.drop(columns=[target_column])
# X_train = X_train.drop(columns=['loudness'])
X_train = X_train.drop(columns = ['Unnamed: 0'])
y_train, unique_labels = pd.factorize(train[target_column])


categorical_cols = X_train.select_dtypes(include=['object', 'category','bool']).columns.tolist()
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(categorical_cols)
print(numerical_cols)

X_train = X_train[numerical_cols].reset_index(drop=True)
mean = X_train.mean()
std = X_train.std()
X_processed = (X_train - mean) / std
X_train = pd.DataFrame(X_processed, columns=X_train.columns)

X_test = test.drop(columns=[target_column])
# X_test = X_test.drop(columns=['loudness'])
X_test = X_test.drop(columns = ['Unnamed: 0'])
y_test, unique_labels = pd.factorize(test[target_column])


X_test = X_test[numerical_cols].reset_index(drop=True)
# mean = X_test.mean()
# std = X_test.std()
X_processed = (X_test - mean) / std
X_test = pd.DataFrame(X_processed, columns=X_test.columns)


X_val = val.drop(columns=[target_column])
# X_val = X_val.drop(columns=['loudness'])
X_val = X_val.drop(columns = ['Unnamed: 0'])
y_val, unique_labels = pd.factorize(val[target_column])



X_val = X_val[numerical_cols].reset_index(drop=True)
# mean = X_val.mean()
# std = X_val.std()
X_processed = (X_val - mean) / std
X_val = pd.DataFrame(X_processed, columns=X_val.columns)





best_params_small = []
knn = KNNVectorized()





try:
    for k in range(1,24,2):
        for metric in ['manhattan']:
            print(f"Testing k={k}, metric={metric}")
            knn.set_params(k, metric)
            knn.fit(X_train.values, y_train)
            knn.set_batch_size(200)
            y_pred_val = knn.predict(X_val.values)
            print ("testing")
            accuracy = Metrics.accuracy(y_val, y_pred_val)
            # micro_f1 = Metrics.f1_score(y_val,y_pred_val,'micro')
            # macro_f1 = Metrics.f1_score(y_val,y_pred_val,'macro')
            best_params_small.append({'k': k, 'distance_metric': metric, 'accuracy': accuracy})
            print(f"Accuracy: {accuracy:.4f}")
except Exception as e:
    print(f"An error occurred: {e}")

print(best_params_small)


best_params = best_params_small

best_params_sorted = sorted(best_params, key=lambda x: x['accuracy'], reverse=True)


print("Top 10 {k, distance metric} pairs by validation accuracy:")
# print(best_params_sorted)

for i, params in enumerate(best_params_sorted[:10], start=1):
    print(f"Rank {i}: k={params['k']}, distance_metric={params['distance_metric']}, Accuracy={params['accuracy']:.4f}")


ks = range(1,24,2)
accuracies_euclidean = [param['accuracy'] for param in best_params if param['distance_metric'] == 'manhattan']

plt.figure(figsize=(10, 6))
plt.plot(ks, accuracies_euclidean, marker='o', label='Euclidean')
plt.xlabel('k')
plt.ylabel('Validation Accuracy')
plt.title('k vs Validation Accuracy for Euclidean Distance')
plt.xticks(ks)
plt.grid(True)
plt.legend()
plt.show()



