import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

k_gmm = 7
k_kmeans = 5


df = pd.read_feather('../../data/external/word-embeddings.feather')
words = df.iloc[:, 0].values  # The first column contains words
embeddings = df.iloc[:, 1].apply(lambda x: np.array(x)).values  # The second column contains embeddings

embeddings = np.vstack(embeddings)
embeddings = StandardScaler().fit_transform(embeddings)


kmeans = KMeans(n_clusters=k_kmeans, random_state=42)
kmeans_labels = kmeans.fit_predict(embeddings)


gmm = GaussianMixture(n_components=6, random_state=42)
gmm_labels = gmm.fit_predict(embeddings)


kmeans_silhouette = silhouette_score(embeddings, kmeans_labels)
dv_kmeans = davies_bouldin_score(embeddings, kmeans_labels)


gmm_sil = silhouette_score(embeddings, gmm_labels)
davb_gmm = davies_bouldin_score(embeddings, gmm_labels)

print(f"K-means Silhouette Score: {kmeans_silhouette}")
print(f"K-means Davies-Bouldin Index: {dv_kmeans}")

print(f"GMM Silhouette Score: {gmm_sil}")
print(f"GMM Davies-Bouldin Index: {davb_gmm}")

tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(14, 6))

# K-Means Plot with centroids
plt.subplot(1, 2, 1)
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=kmeans_labels, palette='tab10', s=50, alpha=0.6)
plt.title(f'K-means Clusters (k={k_kmeans})\nSilhouette: {kmeans_silhouette:.2f}, Davies-Bouldin: {dv_kmeans:.2f}', fontsize=12)
plt.legend(title='Cluster Label')  # Replacing colorbar with legend

# GMM Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x=embeddings_2d[:, 0], y=embeddings_2d[:, 1], hue=gmm_labels, palette='tab20', s=50, alpha=0.6)
plt.title(f'GMM Clusters (k={7})\nSilhouette: {gmm_sil:.2f}, Davies-Bouldin: {davb_gmm:.2f}', fontsize=12)
plt.legend(title='Cluster Label')  # Replacing colorbar with legend

plt.tight_layout()
plt.show()
