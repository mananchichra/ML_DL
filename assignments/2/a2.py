import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKLearnKMeans
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

print("Shape of embeddings:", embeddings.shape)

kmeans = KMeans(k=3)
kmeans.fit(embeddings)
custom_labels = kmeans.predict(embeddings)
custom_cost = kmeans.getCost(embeddings)

print("Custom KMeans Cluster Labels:", custom_labels)
print("Custom KMeans WCSS:", custom_cost)