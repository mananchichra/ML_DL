# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans as SKLearnKMeans
# import os
# import sys
# from sklearn.decomposition import PCA
# sys.path.append(os.path.abspath('../../models/pca'))
# from sklearn.preprocessing import StandardScaler


# from pca import PCA

# # Load the dataset
# df = pd.read_feather('../../data/external/word-embeddings.feather')

# # Extract words and embeddings
# words = df.iloc[:, 0].values  # The first column contains words
# embeddings = (df.iloc[:, 1:].values)  # The rest of the columns are embeddings  
# embeddings = np.vstack([e[0] for e in embeddings])

#  #Instantiate the PCA class for 2 components
# pca_2d = PCA(n_components=2)
# pca_2d.fit(embeddings)
# embeddings_2d = pca_2d.transform(embeddings)

# # Verify the reduction
# print("Check PCA for 2D:", pca_2d.checkPCA(embeddings))  # Should return True

# # Instantiate the PCA class for 3 components
# pca_3d = PCA(n_components=3)
# pca_3d.fit(embeddings)
# embeddings_3d = pca_3d.transform(embeddings)
# # print("Embeddings 3D dtype:", embeddings_3d.dtype)
# # print("Embeddings 3D array:", embeddings_3d)

# # Remove the checkPCA function calls
# print("Embeddings 2D shape:", embeddings_2d.shape)  # Should be (n_samples, 2)
# print("Embeddings 3D shape:", embeddings_3d.shape)  # Should be (n_samples, 3)


# # Verify the reduction
# print("Check PCA for 3D:", pca_3d.checkPCA(embeddings))  # Should return True

# # Plot the 2D PCA result
# plt.figure(figsize=(8, 6))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='blue', marker='o')
# plt.title('2D PCA')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()

# # Plot the 3D PCA result
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c='red', marker='^')
# ax.set_title('3D PCA')
# ax.set_xlabel('Principal Component 1')
# ax.set_ylabel('Principal Component 2')
# ax.set_zlabel('Principal Component 3')
# plt.show()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SKLearnKMeans
import os
import sys
# from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

sys.path.append(os.path.abspath('../../models/pca'))
from pca import PCA
from sklearn.preprocessing import StandardScaler

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

# Verify the reduction
print("Check PCA for 3D:", pca_3d.checkPCA(embeddings))  # Should return True

# Plot the 2D PCA result with color differentiation for clusters
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=np.random.rand(len(embeddings_2d)), cmap='viridis', alpha=0.7, edgecolor='k')
plt.title('2D PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Random Cluster Color')  # Use a colorbar for differentiation
plt.show()

# Plot the 3D PCA result with interactive rotation and colormap
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Use a color map to show cluster differentiation in 3D space
sc = ax.scatter(embeddings_3d[:, 0], embeddings_3d[:, 1], embeddings_3d[:, 2], c=np.random.rand(len(embeddings_3d)), cmap='coolwarm', alpha=0.8)

ax.set_title('3D PCA')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
fig.colorbar(sc, ax=ax, label='Random Cluster Color')  # Add color bar
plt.show()
