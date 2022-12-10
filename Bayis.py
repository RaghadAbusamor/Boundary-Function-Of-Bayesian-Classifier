# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib inline
# scikit-learn modules
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
# Plotting the classification results
from mlxtend.plotting import plot_decision_regions
# Importing the dataset
dataset = load_breast_cancer() 
# Converting to pandas DataFrame
df = pd.DataFrame(dataset.data, columns = dataset.feature_names)
df['target'] = pd.Series(dataset.target)
print("Total samples in our dataset is: {}".format(df.shape[0]))
# Describe the dataset
df.describe()
# Selecting the features
features = ['mean perimeter', 'mean texture']
x = df[features]
# Target variable
y = df['target']
# Splitting the dataset into the training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 25 )
# Fitting Naive Bayes to the Training set
model = GaussianNB()
model.fit(x_train, y_train)
# Predicting the results
y_pred = model.predict(x_test)
# Confusion matrix
print("Confusion Matrix")
matrix = confusion_matrix(y_test, y_pred)
print(matrix)
# Classification Report
print("\nClassification Report")
report = classification_report(y_test, y_pred)
print(report)
# Accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print('Gaussian NB Classification Accuracy of the model: {:.2f}%'.format(accuracy*100))
# Plotting the decision boundary
plt.figure(figsize=(10,6))
plot_decision_regions(x_test.values, y_test.values, clf = model, legend = 2)
plt.title("Decision boundary using Naive Bayes (Test)")
plt.xlabel("mean_perimeter")
plt.ylabel("mean_texture")
plt.show()