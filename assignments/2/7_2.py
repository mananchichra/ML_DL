import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import os, sys
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath('../../models/gmm'))
from gmm import GMM
# Load the dataset
df = pd.read_feather('../../data/external/word-embeddings.feather')

# Extract words and embeddings
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
    
    
kgmm1 = 2
k2 = 3
kgmm3 = 7
k_values = [2,3,7]
results = []

# for k in k_values:
#     gmm = GMM(n_components=k, n_iterations=100, reg_covar=1e-3)
#     gmm.fit(embeddings)
    
#     results.append({
#         'n_components': k,
#         'Coherence': gmm.compute_coherence(embeddings)
#     })

# for i in results:
#     print(i)


# best_model_coherence = max(results, key=lambda x: x['Coherence'])
# print(f"Best model based on coherence: {best_model_coherence}")
def plot_gmm_clusters(X, k_values):
    # Perform PCA to reduce the dimensionality of embeddings to 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 5), sharex=True, sharey=True)
    fig.suptitle('GMM Cluster Plots for Different k Values', fontsize=16)
    
    for idx, k in enumerate(k_values):
        gmm = GMM(n_components=k, n_iterations=100, reg_covar=1e-3)
        gmm.fit(X)
        labels = gmm.predict(X)  # Get cluster labels
        
        axes[idx].scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', s=50, alpha=0.7)
        axes[idx].set_title(f'k={k}')
    
    plt.show()

# Evaluate GMM clustering
for k in k_values:
    gmm = GMM(n_components=k, n_iterations=100, reg_covar=1e-3)
    gmm.fit(embeddings)
    coherence = gmm.compute_coherence(embeddings)
    
    results.append({
        'n_components': k,
        'Coherence': coherence
    })

# Print the results
for i in results:
    print(i)

best_model_coherence = max(results, key=lambda x: x['Coherence'])
print(f"Best model based on coherence: {best_model_coherence}")

# Plot GMM clusters
plot_gmm_clusters(embeddings, k_values)