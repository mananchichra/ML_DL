import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.preprocessing import StandardScaler

df = pd.read_feather('../../data/external/word-embeddings.feather')

words = df.iloc[:, 0].values
embeddings = np.stack(df.iloc[:, 1].apply(lambda x: np.array(x)).values)

scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

linkage_methods = ['ward', 'complete', 'average', 'single']

for method in linkage_methods:
    Z = linkage(embeddings, method=method, metric='euclidean')  # Euclidean distance
    plt.figure(figsize=(10, 7))
    dendrogram(Z)
    plt.title(f'Dendrogram with {method} linkage')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()


best_method = 'ward'  # Example; replace with your choice

Z_best = linkage(embeddings, method=best_method, metric='euclidean')

kbest1 = 5  # Replace with the best k from K-Means clustering
kbest2 = 5  # Replace with the best k from GMM clustering

clusters_kbest1 = fcluster(Z_best, t=kbest1, criterion='maxclust')
clusters_kbest2 = fcluster(Z_best, t=kbest2, criterion='maxclust')


from sklearn.metrics import adjusted_rand_score

ari_kmeans = adjusted_rand_score(kmeans_labels, clusters_kbest1)
print(f'ARI with K-Means: {ari_kmeans}')

ari_gmm = adjusted_rand_score(gmm_labels, clusters_kbest2)
print(f'ARI with GMM: {ari_gmm}')


