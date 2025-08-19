import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKLearnKMeans
import os
import sys
# from sklearn.decomposition import PCA
sys.path.append(os.path.abspath('../../models/pca'))
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity


from pca import PCA

sys.path.append(os.path.abspath('../../models/k-means'))
from k_means import KMeans


# Load the dataset
df = pd.read_feather('../../data/external/word-embeddings.feather')

# Extract words and embeddings
words = df.iloc[:, 0].values  # The first column contains words
embeddings = df.iloc[:, 1].apply(lambda x: np.array(x)).values  # The second column contains embeddings


embeddings = np.vstack(embeddings)
embeddings = StandardScaler().fit_transform(embeddings)

pca_2d = PCA(n_components=2)
pca_2d.fit(embeddings)
embeddings_2d = pca_2d.transform(embeddings)

k2 = 3  # Adjust based on your visualization or estimation

kmeans = KMeans(k=k2, random_state=42)
kmeans.fit(embeddings_2d)
clusters = kmeans.predict(embeddings_2d)

plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='viridis', marker='o')
plt.title('2D PCA with K-means Clusters')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()



pca_full = PCA(n_components=min(10, embeddings.shape[1]))
pca_full.fit(embeddings)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca_full.explained_variance_ratio) + 1), pca_full.explained_variance_ratio, marker='o')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.show()


optimal_dims = 5  # Adjust based on the scree plot

pca_reduced = PCA(n_components=optimal_dims)
pca_reduced.fit(embeddings)
reduced_embeddings = pca_reduced.transform(embeddings)


wcss = []  # Within-cluster sum of squares
for i in range(1, 21):
    kmeans = KMeans(k=i, random_state=42)
    kmeans.fit(reduced_embeddings)
    wcss.append(kmeans.getCost(reduced_embeddings))

plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), wcss, marker='x')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kkmeans3 = 5  


# Apply K-means clustering with the optimal number of clusters
kmeans_reduced = KMeans(k=kkmeans3, random_state=42)
kmeans_reduced.fit(reduced_embeddings)
clusters_reduced = kmeans_reduced.predict(reduced_embeddings)

# Visualize the clusters in 2D (using the first two components of the reduced data)
plt.figure(figsize=(10, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters_reduced, cmap='viridis', marker='o')
# plt.title('Reduced PCA with K-means Clusters')
plt.title('K-means Clusters (Kkmean1)')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster Label')
plt.show()

