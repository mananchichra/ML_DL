import numpy as np
import matplotlib.pyplot as plt

class KernelDensityEstimator:
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        """
        Initialize the KDE with a specific bandwidth and kernel type.
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.data = None

    def fit(self, data):
        """
        Store the data for KDE estimation.
        """
        self.data = data

    def _gaussian_kernel(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def _box_kernel(self, u):
        return 0.5 * (np.abs(u) <= 1)

    def _triangular_kernel(self, u):
        return (1 - np.abs(u)) * (np.abs(u) <= 1)

    def _compute_kernel(self, u):
        if self.kernel == 'gaussian':
            return self._gaussian_kernel(u)
        elif self.kernel == 'box':
            return self._box_kernel(u)
        elif self.kernel == 'triangular':
            return self._triangular_kernel(u)
        else:
            raise ValueError("Unknown kernel type.")

    def predict(self, x):
        """
        Estimate the density at a given point x using the KDE.
        """
        if self.data is None:
            raise ValueError("The KDE model must be fit before predicting.")

        distances = np.linalg.norm((self.data - x) / self.bandwidth, axis=1)
        kernel_values = self._compute_kernel(distances)
        density = kernel_values.mean() / self.bandwidth**self.data.shape[1]
        return density

    def visualize(self, grid_size=100):
        """
        Visualize the KDE density over a 2D grid.
        """
        if self.data is None:
            raise ValueError("The KDE model must be fit before visualization.")

        x_min, x_max = self.data[:, 0].min() - 1, self.data[:, 0].max() + 1
        y_min, y_max = self.data[:, 1].min() - 1, self.data[:, 1].max() + 1

        x_grid = np.linspace(x_min, x_max, grid_size)
        y_grid = np.linspace(y_min, y_max, grid_size)
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)

        grid_points = np.c_[x_mesh.ravel(), y_mesh.ravel()]
        densities = np.array([self.predict(point) for point in grid_points]).reshape(grid_size, grid_size)

        plt.contourf(x_mesh, y_mesh, densities, levels=20, cmap='viridis')
        plt.scatter(self.data[:, 0], self.data[:, 1], s=5, color='black')
        plt.title(f"KDE - {self.kernel.capitalize()} Kernel")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.colorbar(label="Density")
        plt.show()

