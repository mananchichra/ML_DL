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
# Instantiate the custom GMM model
gmm_custom = GMM(n_components=3, n_iterations=1000, reg_covar=1e-6)  # Include n_iterations and reg_covar if needed
gmm_custom.fit(embeddings)

# Get the parameters of the custom GMM model
params = gmm_custom.get_params()

# Instantiate sklearn's GMM model
gmm_sklearn = GaussianMixture(n_components=3, random_state=42)
gmm_sklearn.fit(embeddings)

# Get the cluster assignments and log-likelihood from sklearn GMM
sklearn_labels = gmm_sklearn.predict(embeddings)
sklearn_log_likelihood = gmm_sklearn.score_samples(embeddings).sum()

# Get the log-likelihood from custom GMM
custom_log_likelihood = gmm_custom.get_likelihood(embeddings)  # Fix method call to match class definition

# Compare log-likelihoods
print(f"Custom GMM Log-Likelihood: {custom_log_likelihood}")
print(f"Sklearn GMM Log-Likelihood: {sklearn_log_likelihood}")

# Dimensionality reduction for visualization (PCA to 2D)
# pca = PCA(n_components=2)
# embeddings_2d = pca.fit_transform(embeddings)
embeddings_2d = embeddings
# Plot the clusters for custom GMM
plt.figure(figsize=(12, 6))

# Custom GMM clusters
plt.subplot(1, 2, 1)
custom_labels = np.argmax(gmm_custom.get_membership(embeddings), axis=1)  # Get cluster labels from membership matrix
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=custom_labels, cmap='viridis', s=30)
plt.title('(k2 = 3)Clusters by Custom GMM')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# Sklearn GMM clusters
plt.subplot(1, 2, 2)
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=sklearn_labels, cmap='viridis', s=30)
plt.title('(K2 = 3)Clusters by Sklearn GMM')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

# Show the plots
plt.tight_layout()
plt.show()
