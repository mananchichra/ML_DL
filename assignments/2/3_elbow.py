import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.abspath('../../models/k-means'))

from k_means import KMeans

df = pd.read_feather('../../data/external/word-embeddings.feather')

words = df.iloc[:, 0].values
embeddings = np.stack(df.iloc[:, 1].apply(lambda x: np.array(x)).values)

wcss = []
kvalues = range(1, 11)  

for k in kvalues:
    kmeans = KMeans(k=k,random_state=47)
    kmeans.fit(embeddings)
    wcss.append(kmeans.getCost(embeddings))

plt.figure(figsize=(10, 6))
plt.plot(kvalues, wcss, marker='o', linestyle='dashdot')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.grid(True)
plt.show()

kkmeans1 = 6  

optimal_kmeans = KMeans(k=kkmeans1)
optimal_kmeans.fit(embeddings)
optimal_cost = optimal_kmeans.getCost(embeddings)

print(f"Optimal KMeans WCSS with k={kkmeans1}:", optimal_cost)