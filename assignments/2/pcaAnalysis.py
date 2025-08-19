import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKLearnKMeans
import os
import sys
from sklearn.decomposition import PCA
sys.path.append(os.path.abspath('../../models/pca'))
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity


from pca import PCA

# Load the dataset
df = pd.read_feather('../../data/external/word-embeddings.feather')

# Extract words and embeddings
words = df.iloc[:, 0].values  # The first column contains words
embeddings = (df.iloc[:, 1:].values)  # The rest of the columns are embeddings
embeddings = np.vstack([e[0] for e in embeddings])


# Instantiate the PCA class for 2 components
pca_2d = PCA(n_components=2)
pca_2d.fit(embeddings)
embeddings_2d = pca_2d.transform(embeddings)

# Verify the reduction
print("Check PCA for 2D:", pca_2d.checkPCA(embeddings))  # Should return True

# Instantiate the PCA class for 3 components
pca_3d = PCA(n_components=3)
pca_3d.fit(embeddings)
embeddings_3d = pca_3d.transform(embeddings)
# print("Embeddings 3D dtype:", embeddings_3d.dtype)
# print("Embeddings 3D array:", embeddings_3d)

# Verify the reduction
print("Check PCA for 3D:", pca_3d.checkPCA(embeddings))  # Should return True

# Color coding based on proximity to origin in 2D
dist_2d = np.linalg.norm(embeddings_2d, axis=1)
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=dist_2d, cmap='viridis', marker='o')
plt.colorbar(label='Distance from Origin')
plt.title('2D PCA with Proximity Color Coding')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# 3D Plot with Proximity Color Coding
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
dist_3d = np.linalg.norm(embeddings_3d, axis=1)
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=dist_3d, cmap='plasma', marker='^')
ax.set_title('3D PCA with Proximity Color Coding')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
fig.colorbar(scatter, ax=ax, label='Distance from Origin')
plt.show()

# Adding a density plot for the 2D PCA result
kde = KernelDensity(bandwidth=0.5).fit(embeddings_2d)
density_2d = kde.score_samples(embeddings_2d)
plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=density_2d, cmap='coolwarm', marker='o')
plt.colorbar(label='Density')
plt.title('2D PCA with Density Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

# Adding a density plot for the 3D PCA result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=density_2d, cmap='coolwarm', marker='^')
ax.set_title('3D PCA with Density Plot')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
fig.colorbar(scatter, ax=ax, label='Density')
plt.show()

# Estimate the approximate number of clusters based on visual inspection
# Manual inspection and estimation of clusters (k2)
k2_estimated = 5  # Replace this with your estimation based on the plots
print(f"Estimated number of clusters (k2): {k2_estimated}") 