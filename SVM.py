# Description:
# This script demonstrates the use of Support Vector Machines (SVM) for classifying apples and oranges based on their weights and sizes. It prepares the data, trains an SVM classifier, and visualizes the decision boundary on both the training and test datasets.

# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

# Function to prepare data by extracting features and labels
def prepare_data(data):
    X = data.iloc[:, 0:2].values
    y = data.iloc[:, 2].values
    lb = LabelEncoder()
    y = lb.fit_transform(y)
    return X, y

# Function to visualize the decision boundary and data points
def visualize_data(X, y, classifier, title):
    plt.figure(figsize=(7, 7))
    x1, x2 = np.meshgrid(np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01),
                        np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01))
    plt.contourf(x1, x2, classifier.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
                 alpha=0.75, cmap=ListedColormap(('black', 'white')))
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())
    for i, j in enumerate(np.unique(y)):
        plt.scatter(X[y == j, 0], X[y == j, 1],
                    c=['red', 'orange'][i],
                    label=j)
    plt.title(title)
    plt.xlabel('Weights In Grams')
    plt.ylabel('Size In cms')
    plt.legend()
    plt.show()

# Function to train the SVM classifier and visualize the results
def train_and_visualize(train_file, test_file, title):
    # Loading training and test datasets
    training_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    # Extracting features and labels from the datasets
    x_train, y_train = prepare_data(training_data)
    x_test, y_test = prepare_data(test_data)

    # Initializing SVM classifier with radial basis function kernel
    classifier = SVC(kernel='rbf', random_state=1, C=1, gamma='auto')
    classifier.fit(x_train, y_train)

    # Visualizing the decision boundary on training and test datasets
    visualize_data(x_train, y_train, classifier, f'{title} (Training Data)')
    visualize_data(x_test, y_test, classifier, f'{title} Predictions')

# Training and visualizing SVM for classifying apples and oranges
train_and_visualize('T.csv', 'T1.csv', 'Apples Vs Oranges')
