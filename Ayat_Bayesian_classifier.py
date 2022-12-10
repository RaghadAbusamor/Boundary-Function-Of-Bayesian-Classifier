import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
data = pd.read_csv("iris.csv")
x1 = data["x1"]
x2 = data["x2"]
class_labels = data["variety"]

# computer mean and covariance
mean0 = np.mean(data[class_labels == 0][["x1", "x2"]], axis=0)
cov0 = np.cov(data[class_labels == 0][["x1", "x2"]], rowvar=False)
mean1 = np.mean(data[class_labels == 1][["x1", "x2"]], axis=0)
cov1 = np.cov(data[class_labels == 1][["x1", "x2"]], rowvar=False)

# Compute the boundary function and the boundary points
def boundary_func(x, mean0, cov0, mean1, cov1):
    # Compute the inverse of the covariance matrix
    inv_cov0 = np.linalg.inv(cov0)
    inv_cov1 = np.linalg.inv(cov1)
    f = np.dot(mean0.T, np.dot(inv_cov0, x)) - np.dot(mean1.T, np.dot(inv_cov1, x)) + 0.5 * (np.log(np.linalg.det(cov0)) - np.log(np.linalg.det(cov1)))
    return f

# Compute the boundary points
x1_min, x1_max = min(x1), max(x1)
x2_min, x2_max = min(x2), max(x2)
x1_range = np.linspace(x1_min, x1_max, num=100)
x2_range = np.linspace(x2_min, x2_max, num=100)
X1, X2 = np.meshgrid(x1_range, x2_range)
F = boundary_func(np.array([X1.flatten(), X2.flatten()]), mean0, cov0, mean1, cov1)
plt.contour(X1, X2, F.reshape(X1.shape), levels=[0], colors="red")
plt.scatter(x1, x2, c=class_labels)
plt.show()