# import numpy as np

# class KMeans:
#     def __init__(self, k=3, max_iters=100, tol=1e-5, random_state=None):
#         self.k = k
#         self.max_iters = max_iters
#         self.tol = tol
#         self.centroids = None
#         self.labels = None
#         self.random_state = random_state
    
#     def fit(self, X):
#         # Step 1: Initialize centroids randomly
#         if self.random_state is not None:
#             np.random.seed(self.random_state)

            
#         self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
#         for i in range(self.max_iters):
#             # Step 2: Expectation Step - Assign clusters
#             self.labels = self._assign_clusters(X)
            
#             # Step 3: Maximization Step - Recompute centroids
#             new_centroids = self._compute_centroids(X)
            
#             # Step 4: Check for convergence
#             if np.linalg.norm(new_centroids - self.centroids) < self.tol:
#                 break
#             self.centroids = new_centroids
    
#     def _assign_clusters(self, X):
#         # Calculate the distance between each point and each centroid
#         distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
#         # Assign each point to the closest centroid
#         return np.argmin(distances, axis=1)
    
#     def _compute_centroids(self, X):
#         # Recompute centroids as the mean of all points assigned to each cluster
#         new_centroids = np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
#         return new_centroids
    
#     def predict(self, X):
#         return self._assign_clusters(X)
    
#     def getCost(self,embeddings):
#         # doing (WCSS) here
#         distances = np.linalg.norm(embeddings - self.centroids[self.labels], axis=1)
#         return np.sum(distances**2)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class KMeans:
    def __init__(self, k=3, max_iters=100, tol=1e-5, random_state=None, distance_metric='euclidean'):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None
        self.random_state = random_state
        self.distance_metric = distance_metric
    
    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if X.shape[0] < self.k:
            raise ValueError("Number of data points must be at least equal to the number of clusters")

        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
        
        for i in range(self.max_iters):
            self.labels = self._assign_clusters(X)
            new_centroids = self._compute_centroids(X)
            
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
    
    def _assign_clusters(self, X):
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)
    
    def _compute_distances(self, X):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        elif self.distance_metric == 'manhattan':
            return np.abs(X[:, np.newaxis] - self.centroids).sum(axis=2)
        else:
            raise ValueError("Unsupported distance metric")
    
    def _compute_centroids(self, X):
        return np.array([X[self.labels == i].mean(axis=0) for i in range(self.k)])
    
    def predict(self, X):
        return self._assign_clusters(X)
    
    def getCost(self, embeddings):
        distances = np.linalg.norm(embeddings - self.centroids[self.labels], axis=1)
        return np.sum(distances**2)

    def compute_coherence(self, X):
        """
        Compute coherence of clusters using average pairwise cosine similarity within each cluster.
        
        Parameters:
        - X: Data to compute coherence
        
        Returns:
        - coherence_scores: Coherence scores for each cluster
        """
        coherence_scores = []
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) < 2:
                coherence_scores.append(0)
                continue
            similarity_matrix = cosine_similarity(cluster_points)
            np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
            coherence = np.mean(similarity_matrix)
            coherence_scores.append(coherence)
        return coherence_scores

# Example usage:
# def evaluate_clustering(X, k_values):
#     for k in k_values:
#         kmeans = KMeans(k=k, random_state=42)
#         kmeans.fit(X)
#         coherence_scores = kmeans.compute_coherence(X)
#         avg_coherence = np.mean(coherence_scores)
#         print(f'k={k}, Average Coherence: {avg_coherence:.4f}')

# Assuming `embeddings` is your data matrix and `k_values` is a list of k values to evaluate
# Example: k_values = [2, 3, 4, 5]
# evaluate_clustering(embeddings, k_values)
