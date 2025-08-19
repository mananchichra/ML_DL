import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath('../../models/gmm'))
from gmm import GMM
# Load the dataset
df = pd.read_feather('../../data/external/word-embeddings.feather')



words = df.iloc[:, 0].values  # The first column contains words
embeddings = np.stack(df.iloc[:, 1].apply(lambda x: np.array(x)).values)  # Extract embeddings

# Check for NaN or Inf values in the embeddings
if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
    raise ValueError("Data contains NaN or Inf values. Please clean the data.")

# Optionally, replace NaN or Inf values with a specific value
embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

# Standardize the embeddings
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
    print("NaN/Inf found after scaling!")
    embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)

pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(embeddings)
# embeddings_pca = embeddings

# Define range for the number of clusters
n_clusters_range = range(1, 11)  # Test GMMs with 1 to 10 clusters
bic_scores = []
aic_scores = []

# Fit GMM models and calculate BIC and AIC for each
for n_clusters in n_clusters_range:
    gmm = GMM(n_components=n_clusters,n_iterations=100, reg_covar=1e-6)
    gmm.fit(embeddings_pca)
    bic_scores.append(gmm.bic(embeddings_pca))
    aic_scores.append(gmm.aic(embeddings_pca))

# Plot BIC and AIC scores
plt.figure(figsize=(10, 6))
plt.plot(n_clusters_range, bic_scores, label='BIC', marker='o')
plt.plot(n_clusters_range, aic_scores, label='AIC', marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('BIC and AIC Scores for GMM with Different Number of Clusters')
plt.legend()
plt.show()

# Find the optimal number of clusters based on minimum BIC or AIC
optimal_clusters_bic = n_clusters_range[np.argmin(bic_scores)]
optimal_clusters_aic = n_clusters_range[np.argmin(aic_scores)]

print(f"Optimal number of clusters based on BIC: {optimal_clusters_bic}")
print(f"Optimal number of clusters based on AIC: {optimal_clusters_aic}")


kgmm3 = 7  # or you can use optimal_clusters_aic

gmm_optimal = GMM(n_components=kgmm3, n_iterations=100, reg_covar=1e-6)
# gmm_optimal = GMM(n_components=kgmm1, random_state = 42)
gmm_optimal.fit(embeddings_pca)

# Predict cluster labels for the embeddings
cluster_labels = gmm_optimal.predict(embeddings_pca)

# Get GMM parameters
means = gmm_optimal.means_
covariances = gmm_optimal.covariances_
weights = gmm_optimal.weights_

# Print the results
print(f"GMM with {kgmm3} = 7 clusters fitted successfully.")
print(f"Means: {means}")
print(f"Covariances: {covariances}")
print(f"Weights: {weights}")

#plot the data
plt.figure(figsize=(10, 6))
plt.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], c=cluster_labels, cmap='viridis', edgecolor='k', s=150)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('Word Embeddings with GMM Clustering')
plt.grid(True)
plt.show()