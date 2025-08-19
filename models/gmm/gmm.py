
# import numpy as np
# from scipy.stats import multivariate_normal

# class GMM:
#     def __init__(self, n_components, n_iterations, tol=1e-4, reg_covar=1e-3):
#         self.n_components = n_components
#         self.n_iterations = n_iterations
#         self.tol = tol
#         self.reg_covar = reg_covar  # Regularization term for covariance
#         self.weights_ = None
#         self.means_ = None
#         self.covariances_ = None
#         self.converged_ = False

#     def _initialize_parameters(self, X):
#         n_samples, n_features = X.shape
#         random_indices = np.random.choice(n_samples, self.n_components, replace=False)
#         self.means_ = X[random_indices]
#         self.weights_ = np.full(self.n_components, 1 / self.n_components)
#         self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

#     def fit(self, X):
#         self._initialize_parameters(X)
#         n_samples = X.shape[0]
        
#         for i in range(self.n_iterations):
#             # E-step
#             responsibilities = self._e_step(X)
            
#             # Debugging: Check for NaN/Inf in responsibilities
#             if np.any(np.isnan(responsibilities)) or np.any(np.isinf(responsibilities)):
#                 print(f"NaN/Inf detected in responsibilities at iteration {i}")
#                 print(f"Responsibilities:\n{responsibilities}")
#                 break
            
#             # M-step
#             self._m_step(X, responsibilities)
            
#             # Check convergence
#             if i > 0 and np.allclose(prev_responsibilities, responsibilities, atol=self.tol):
#                 self.converged_ = True
#                 break
#             prev_responsibilities = responsibilities

#     def _e_step(self, X):
#         log_likelihoods = np.array([
#             np.log(self.weights_[j]) + multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covariances_[j])
#             for j in range(self.n_components)
#         ]).T

#         # Stabilize by subtracting max log-likelihood to prevent overflow in exp
#         max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
#         log_likelihoods -= max_log_likelihood
        
#         responsibilities = np.exp(log_likelihoods)
#         total_likelihood = np.sum(responsibilities, axis=1, keepdims=True)

#         # Prevent division by zero
#         total_likelihood = np.where(total_likelihood == 0, 1e-6, total_likelihood)

#         return responsibilities / total_likelihood

#     def _m_step(self, X, responsibilities):
#         n_samples = X.shape[0]
#         for j in range(self.n_components):
#             responsibility = responsibilities[:, j]
#             total_responsibility = responsibility.sum()
#             self.means_[j] = np.dot(responsibility, X) / total_responsibility
#             diff = X - self.means_[j]
#             self.covariances_[j] = np.dot(responsibility * diff.T, diff) / total_responsibility + self.reg_covar * np.eye(X.shape[1])
#             self.weights_[j] = total_responsibility / n_samples

#     def get_params(self):
#         return {'weights': self.weights_, 'means': self.means_, 'covariances': self.covariances_}

#     def get_membership(self, X):
#         return self._e_step(X)

#     def get_likelihood(self, X):
#         log_likelihoods = np.array([
#             np.log(self.weights_[j]) + multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covariances_[j])
#             for j in range(self.n_components)
#         ]).T
        
#         # Use log-sum-exp trick for numerical stability
#         max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
#         log_likelihoods -= max_log_likelihood
        
#         return np.sum(np.log(np.sum(np.exp(log_likelihoods), axis=1)) + max_log_likelihood.flatten())

#     def aic(self, X):
#         n_samples, n_features = X.shape
#         log_likelihood = self.get_likelihood(X)
#         # Number of parameters: means + covariances + weights
#         n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
#         aic = 2 * n_params - 2 * log_likelihood
#         return aic

#     def bic(self, X):
#         n_samples, n_features = X.shape
#         log_likelihood = self.get_likelihood(X)
#         # Number of parameters: means + covariances + weights
#         n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
#         bic = np.log(n_samples) * n_params - 2 * log_likelihood
#         return bic
#     def predict(self, X):
#         responsibilities = self._e_step(X)
#         return np.argmax(responsibilities, axis=1)
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics.pairwise import cosine_similarity

class GMM:
    def __init__(self, n_components, n_iterations, tol=1e-4, reg_covar=1e-3):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol
        self.reg_covar = reg_covar  # Regularization term for covariance
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None
        self.converged_ = False

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape
        random_indices = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_indices]
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def fit(self, X):
        self._initialize_parameters(X)
        n_samples = X.shape[0]
        
        for i in range(self.n_iterations):
            # E-step
            responsibilities = self._e_step(X)
            
            # Debugging: Check for NaN/Inf in responsibilities
            if np.any(np.isnan(responsibilities)) or np.any(np.isinf(responsibilities)):
                print(f"NaN/Inf detected in responsibilities at iteration {i}")
                print(f"Responsibilities:\n{responsibilities}")
                break
            
            # M-step
            self._m_step(X, responsibilities)
            
            # Check convergence
            if i > 0 and np.allclose(prev_responsibilities, responsibilities, atol=self.tol):
                self.converged_ = True
                break
            prev_responsibilities = responsibilities

    def _e_step(self, X):
        log_likelihoods = np.array([
            np.log(self.weights_[j]) + multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covariances_[j])
            for j in range(self.n_components)
        ]).T

        # Stabilize by subtracting max log-likelihood to prevent overflow in exp
        max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
        log_likelihoods -= max_log_likelihood
        
        responsibilities = np.exp(log_likelihoods)
        total_likelihood = np.sum(responsibilities, axis=1, keepdims=True)

        # Prevent division by zero
        total_likelihood = np.where(total_likelihood == 0, 1e-6, total_likelihood)

        return responsibilities / total_likelihood

    def _m_step(self, X, responsibilities):
        n_samples = X.shape[0]
        for j in range(self.n_components):
            responsibility = responsibilities[:, j]
            total_responsibility = responsibility.sum()
            self.means_[j] = np.dot(responsibility, X) / total_responsibility
            diff = X - self.means_[j]
            self.covariances_[j] = np.dot(responsibility * diff.T, diff) / total_responsibility + self.reg_covar * np.eye(X.shape[1])
            self.weights_[j] = total_responsibility / n_samples

    def get_params(self):
        return {'weights': self.weights_, 'means': self.means_, 'covariances': self.covariances_}

    def get_membership(self, X):
        return self._e_step(X)

    def get_likelihood(self, X):
        log_likelihoods = np.array([
            np.log(self.weights_[j]) + multivariate_normal.logpdf(X, mean=self.means_[j], cov=self.covariances_[j])
            for j in range(self.n_components)
        ]).T
        
        # Use log-sum-exp trick for numerical stability
        max_log_likelihood = np.max(log_likelihoods, axis=1, keepdims=True)
        log_likelihoods -= max_log_likelihood
        
        return np.sum(np.log(np.sum(np.exp(log_likelihoods), axis=1)) + max_log_likelihood.flatten())

    def aic(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.get_likelihood(X)
        # Number of parameters: means + covariances + weights
        n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
        aic = 2 * n_params - 2 * log_likelihood
        return aic

    def bic(self, X):
        n_samples, n_features = X.shape
        log_likelihood = self.get_likelihood(X)
        # Number of parameters: means + covariances + weights
        n_params = (self.n_components * n_features) + (self.n_components * (n_features * (n_features + 1)) / 2) + (self.n_components - 1)
        bic = np.log(n_samples) * n_params - 2 * log_likelihood
        return bic

    def predict(self, X):
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def compute_coherence(self, X):
        """Compute coherence of the clusters."""
        labels = self.predict(X)
        n_clusters = np.max(labels) + 1
        coherence_scores = []

        for i in range(n_clusters):
            cluster_points = X[labels == i]
            if cluster_points.shape[0] > 1:
                # Compute pairwise cosine similarity within the cluster
                similarities = cosine_similarity(cluster_points)
                # Average similarity for all pairs in the cluster
                avg_similarity = np.mean(similarities[np.triu_indices(len(cluster_points), k=1)])
                coherence_scores.append(avg_similarity)

        return np.mean(coherence_scores) if coherence_scores else 0

# Example usage:
# np.random.seed(42)
# X = np.random.randn(100, 512)  # Example high-dimensional data
# gmm_custom = GMM(n_components=5, n_iterations=100, reg_covar=1e-3)
# gmm_custom.fit(X)
# print("Converged:", gmm_custom.converged_)
# print("Log-Likelihood:", gmm_custom.get_likelihood(X))
# print("AIC:", gmm_custom.aic(X))
# print("BIC:", gmm_custom.bic(X))
# print("Coherence:", gmm_custom.compute_coherence(X))
