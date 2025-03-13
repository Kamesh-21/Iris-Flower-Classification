#!/usr/bin/env python
# coding: utf-8

# # Load the Dataset


# Using Scikit-learn’s Built-in Dataset
from sklearn.datasets import load_iris
import pandas as pd

# Load the iris dataset
iris = load_iris()

# Convert to a DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column (species)
df['species'] = iris.target

# Map target numbers to species names
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display first few rows
df.head()


# # Exploratory Data Analysis

import matplotlib.pyplot as plt
import seaborn as sns

# Check dataset info
print(df.info())

# Summary statistics
print(df.describe())

# Visualize pairwise relationships
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()


# ## Prepare Data for Training


# Separate Features & Labels
X = df.drop(columns=['species'])  # Features (measurements)
y = df['species']  # Labels (flower species)




# Split Data into Training & Testing Sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Train a Machine Learning Model



from sklearn.linear_model import LogisticRegression

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)


# ## Evaluate Model Performance


from sklearn.metrics import accuracy_score, classification_report

# Predict on test data
y_pred = model.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification report
print(classification_report(y_test, y_pred))


# # Make Predictions on New Data



# Test the model with new iris measurements.

import pandas as pd

# Create a DataFrame with the same feature names as the training data
new_sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X_train.columns)

# Predict the species
predicted_species = model.predict(new_sample)
print("Predicted Species:", predicted_species[0])



# ## Visualize Results



from sklearn.svm import SVC

# Train SVM model for visualization
svm_model = SVC(kernel='linear')
svm_model.fit(X_train[['sepal length (cm)', 'sepal width (cm)']], y_train)

# Create a scatter plot
sns.scatterplot(x=X_train['sepal length (cm)'], y=X_train['sepal width (cm)'], hue=y_train)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Flower Classification Decision Boundary")
plt.show()






