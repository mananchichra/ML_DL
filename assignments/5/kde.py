import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/knn')
# Function to generate uniform points within a circle
def generate_points_within_circle(num_points, radius, center=(0, 0)):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))  # sqrt ensures uniform distribution
    x_points = radii * np.cos(angles) + center[0]
    y_points = radii * np.sin(angles) + center[1]
    return x_points, y_points

# Parameters for the larger circle
num_points_large = 3000
radius_large = 3

# Generate points for the larger circle
x_large, y_large = generate_points_within_circle(num_points_large, radius_large)

# Parameters for the smaller circle
num_points_small = 500
radius_small = 0.5
center_small = (1, 1)

# Generate points for the smaller circle
x_small, y_small = generate_points_within_circle(num_points_small, radius_small, center_small)

# Combine all points
x = np.hstack((x_large, x_small))
y = np.hstack((y_large, y_small))

# Plotting
plt.figure(figsize=(6, 6))
plt.scatter(x, y, s=1, color='black')
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.title("Synthetic Dataset with Overlapping Density Regions")
plt.grid(True)
plt.show()


from KDE import KernelDensityEstimator



# Generate synthetic dataset
def generate_circle_points(num_points, radius, center=(0, 0)):
    angles = np.random.uniform(0, 2 * np.pi, num_points)
    radii = radius * np.sqrt(np.random.uniform(0, 1, num_points))
    x = radii * np.cos(angles) + center[0]
    y = radii * np.sin(angles) + center[1]
    return x, y

x_large, y_large = generate_circle_points(3000, 3)
x_small, y_small = generate_circle_points(500, 0.5, center=(1, 1))

x = np.hstack((x_large, x_small))
y = np.hstack((y_large, y_small))
data = np.column_stack((x, y))

# Instantiate and fit KDE estimator
kde = KernelDensityEstimator(bandwidth=0.5, kernel='gaussian')
kde.fit(data)

# Visualize KDE density
kde.visualize(grid_size=100)


sys.path.insert(0, '/home/mananchichra/Downloads/SMAI_ASSIGNMENT/models/gmm')
from GMM import gmm



# Generate synthetic dataset
def generate_synthetic_data():
    num_points_large = 3000
    num_points_small = 500
    radius_large = 3
    radius_small = 0.5
    center_small = (1, 1)

    angles_large = np.random.uniform(0, 2 * np.pi, num_points_large)
    radii_large = radius_large * np.sqrt(np.random.uniform(0, 1, num_points_large))
    x_large = radii_large * np.cos(angles_large)
    y_large = radii_large * np.sin(angles_large)

    angles_small = np.random.uniform(0, 2 * np.pi, num_points_small)
    radii_small = radius_small * np.sqrt(np.random.uniform(0, 1, num_points_small))
    x_small = radii_small * np.cos(angles_small) + center_small[0]
    y_small = radii_small * np.sin(angles_small) + center_small[1]

    x = np.concatenate([x_large, x_small])
    y = np.concatenate([y_large, y_small])
    return np.vstack([x, y]).T


data = generate_synthetic_data()
gmm = gmm(n_components=6, max_iters=100, tol=1e-3)
gmm.fit(data)
gmm.visualize(grid_size=100)
