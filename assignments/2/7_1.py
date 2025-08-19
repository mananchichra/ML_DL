import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import os
import sys
sys.path.append(os.path.abspath('../../models/k-means'))

from k_means import KMeans

# Load the dataset
df = pd.read_feather('../../data/external/word-embeddings.feather')

# Extract words and embeddings
words = df.iloc[:, 0].values  # The first column contains words
embeddings = (df.iloc[:, 1:].values)  # The rest of the columns are embeddings
embeddings = np.vstack([e[0] for e in embeddings])

kmeans1 = 5
k2 = 3
kmeans3 = 6
k_values = [kmeans1,kmeans3,k2]

def evaluate_clustering(X, k_values):
    for k in k_values:
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X)
        coherence_scores = kmeans.compute_coherence(X)
        avg_coherence = np.mean(coherence_scores)
        print(f'k={k}, Average Coherence: {avg_coherence:.4f}')
        print(f'Cluster Coherence Scores: {coherence_scores}')

def plot_clusters(X, k_values):
    # Perform PCA to reduce the dimensionality of embeddings to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle('Cluster Plots for Different k Values', fontsize=16)
    
    for idx, k in enumerate(k_values):
        kmeans = KMeans(k=k, random_state=42)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        
        axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        centers_pca = pca.transform(kmeans.centroids)
        axes[idx].scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', s=200, label='Centroids')
        axes[idx].set_title(f'k={k}')
        axes[idx].legend()

    plt.show()
evaluate_clustering(embeddings, k_values)
# Evaluate clustering
# silhouette_scores = evaluate_clustering(embeddings, k_values)
plot_clusters(embeddings, k_values)