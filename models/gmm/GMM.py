import numpy as np
import matplotlib.pyplot as plt


class gmm:
    def __init__(self, n_components=2, max_iters=100, tol=1e-3):
        """
        Initializes the GMM model with given parameters.
        """
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.means = None
        self.covariances = None
        self.weights = None

    def _initialize(self, data):
        """
        Randomly initializes parameters for the GMM.
        """
        n_samples, n_features = data.shape
        idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = data[idx]
        self.covariances = np.array([np.cov(data, rowvar=False) + np.eye(n_features) * 1e-6] * self.n_components)
        self.weights = np.ones(self.n_components) / self.n_components

    @staticmethod
    def _multivariate_gaussian(x, mean, covariance):
        """
        Computes the multivariate Gaussian density function for a point.
        """
        size = len(mean)
        cov_inv = np.linalg.inv(covariance)
        diff = x - mean
        exp_term = np.exp(-0.5 * np.dot(np.dot(diff.T, cov_inv), diff))
        norm_term = 1 / np.sqrt((2 * np.pi) ** size * np.linalg.det(covariance))
        return norm_term * exp_term

    def _expectation(self, data):
        """
        E-step: Calculate responsibilities.
        """
        n_samples = data.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            for i, point in enumerate(data):
                responsibilities[i, k] = self.weights[k] * self._multivariate_gaussian(
                    point, self.means[k], self.covariances[k]
                )

        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _maximization(self, data, responsibilities):
        """
        M-step: Update weights, means, and covariances.
        """
        n_samples = data.shape[0]
        for k in range(self.n_components):
            Nk = responsibilities[:, k].sum()
            self.weights[k] = Nk / n_samples
            self.means[k] = np.dot(responsibilities[:, k], data) / Nk
            diff = data - self.means[k]
            self.covariances[k] = (responsibilities[:, k][:, None] * diff).T @ diff / Nk

    def fit(self, data):
        """
        Fits the GMM model using the EM algorithm.
        """
        self._initialize(data)
        for iteration in range(self.max_iters):
            old_weights = self.weights.copy()
            responsibilities = self._expectation(data)
            self._maximization(data, responsibilities)

            if np.all(np.abs(self.weights - old_weights) < self.tol):
                break

    def predict_density(self, x):
        """
        Predicts the density of a given point x.
        """
        density = sum(
            self.weights[k] * self._multivariate_gaussian(x, self.means[k], self.covariances[k])
            for k in range(self.n_components)
        )
        return density

    def visualize(self, grid_size=100):
        """
        Visualizes the GMM density in 2D space.
        """
        x_min, x_max = self.means[:, 0].min() - 3, self.means[:, 0].max() + 3
        y_min, y_max = self.means[:, 1].min() - 3, self.means[:, 1].max() + 3
        x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size))
        density_grid = np.zeros_like(x_grid)

        for i in range(grid_size):
            for j in range(grid_size):
                point = np.array([x_grid[i, j], y_grid[i, j]])
                density_grid[i, j] = self.predict_density(point)

        plt.contourf(x_grid, y_grid, density_grid, levels=20, cmap="viridis")
        plt.scatter(self.means[:, 0], self.means[:, 1], color="red", marker="x", s=100, label="Means")
        plt.title("GMM Density Estimation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Density")
        plt.legend()
        plt.show()

