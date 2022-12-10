import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("Iris.csv")
X1 = data["X1"]
X2= data["X2"]
Class = data["class"]
mean_class0 = np.mean(data[Class == 0][["X1", "X2"]], axis=0)
mean_class1 = np.mean(data[Class == 1][["X1", "X2"]], axis=0)
cov_class0 = np.cov(data[Class == 0][["X1", "X2"]], rowvar=False)
cov_class1 = np.cov(data[Class == 1][["X1", "X2"]], rowvar=False)



def boundary_func(x, mean_class0, cov_class0, mean_class1, cov_class1):
    inv_cov_class0 = np.linalg.inv(cov_class0)
    inv_cov_class1 = np.linalg.inv(cov_class1)
    
    f = np.dot(mean_class0.T, np.dot(inv_cov_class0, x)) - np.dot(mean_class1.T, np.dot(inv_cov_class1, x)) + 0.5 * (np.log(np.linalg.det(cov_class0)) - np.log(np.linalg.det(cov_class1)))
    return f

X1_min, X1_max = min(X1), max(X1)
X2_min, X2_max = min(X2), max(X2)
X1_range = np.linspace(X1_min, X1_max, num=100)
X2_range = np.linspace(X2_min, X2_max, num=100)
x1, x2= np.meshgrid(X1_range, X2_range)
F = boundary_func(np.array([x1.flatten(), x2.flatten()]), mean_class0, cov_class0, mean_class1, cov_class1)

#Plot the boundary function 
plt.contour(x1, x2, F.reshape(x1.shape), levels=[0], colors="red")
plt.scatter(X1, X2, c=Class)
plt.show()