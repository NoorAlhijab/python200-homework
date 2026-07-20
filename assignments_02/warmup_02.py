# --- scikit-learn API ---
# scikit-learn Question 1

import numpy as np
from sklearn.linear_model import LinearRegression

years = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

# Create a LinearRegression model
model = LinearRegression()

# Fit model to data
model.fit(years, salary)

# Predict with new data
new_years = np.array([4, 8]).reshape(-1, 1)
predicted_salary = model.predict(new_years)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
print("First predicted salary for 4 years of experience is:", predicted_salary[0])
print("Second predicted salary for 8 years of experience is:", predicted_salary[1])


# scikit-learn Question 2
x = np.array([10, 20, 30, 40, 50])
print("x shape:", x.shape)

# Reshape the array to be two-dimensional
x = x.reshape(-1, 1)
print("x shape after reshaping:", x.shape)

# Scikit-learn expects a two-dimensional array because the NumPy array is one-dimensional.
# With a 1D array, scikit-learn cannot tell whether the values represent samples or features.
# We use reshape(-1, 1) to convert it into a 2D array with one feature.
# The -1 tells NumPy to automatically determine the number of rows based on the length of the array,
# while 1 specifies that there should be one column (one feature).

# scikit-learn Question 3
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)
# Create the model and fit it to the data
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_clusters)
# Predict a label for each point
labels = kmeans.predict(X_clusters)
print("Cluster centers:\n", kmeans.cluster_centers_)
# Count the number of points in each cluster
print("Count of points in each cluster:", np.bincount(labels))

# Create a scatter plot
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap="viridis", s=60, alpha=0.7)
# Add cluster centers to the plot
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="black", marker="X", s=200)
plt.title("KMeans Clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("assignments_02/outputs/kmeans_clusters.png")

# --- Linear Regression ---
# Linear Regression Question 1
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

# Create a scatter plot 
plt.scatter(age, cost, c=smoker, cmap="coolwarm")
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Cost")
plt.savefig("assignments_02/outputs/cost_vs_age.png")

# There are two distinct groups visible.
# Smokers generally have higher medical costs than non-smokers.
# This suggests that the smoker variable has a strong effect on medical cost and is an important feature for predicting cost.

# Linear Regression Question 2
x = age.reshape(-1, 1)
y = cost
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)

# Print the shapes of the training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Linear Regression Question 3
model = LinearRegression()
model.fit(X_train, y_train)
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)
y_pred = model.predict(X_test)

# RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

# R² score
r2 = model.score(X_test, y_test)
print("R² on the test set:", r2)

# The slope shows how much medical cost is expected to increase for each additional year of age.

# Linear Regression Question 4
X_full = np.column_stack([age, smoker])
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y, test_size=0.2, random_state=42)

model_full = LinearRegression()
model_full.fit(X_train_full, y_train_full)
print("age coefficient:    ", model_full.coef_[0])
print("smoker coefficient: ", model_full.coef_[1])
print("R² on the full model:", model_full.score(X_test_full, y_test_full))
# The smoker coefficient represents the increase in predicted medical cost for smokers compared to non-smokers while keeping age constant.

# Linear Regression Question 5
y_pred_full = model_full.predict(X_test_full)

# Create a scatter plot of predicted vs actual medical costs
plt.scatter(y_pred_full, y_test_full)
# Add a diagonal line for reference
plt.plot([y_pred_full.min(), y_pred_full.max()], [y_pred_full.min(), y_pred_full.max()])
plt.title("Predicted vs Actual Cost")
plt.xlabel("Predicted Cost")
plt.ylabel("Actual Cost")
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
# Points above the diagonal line mean the actual cost is higher than the predicted cost, 
# while points below the line mean the actual cost is lower than predicted.