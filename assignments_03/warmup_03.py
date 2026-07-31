import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

# --- Preprocessing --- 
# Preprocessing Question 1
# Split X and y into training and test sets using an 80/20 split with stratify=y and random_state=42
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
#  Print the shapes of all four arrays
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test shape:", y_test.shape)

# Preprocessing Question 2
# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Print the mean of each column
print("Means:", X_train_scaled.mean(axis=0))

# Fit the scaler only on X_train to keep X_test unseen during training.

# --- KNN --- 
# KNN Question 1

# Build KNN model with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)
# Fit using unscaled training data
knn.fit(X_train, y_train)
# Predict on the test data
knn_unscaled_pred = knn.predict(X_test)
# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, knn_unscaled_pred))
print(classification_report(y_test, knn_unscaled_pred))

# KNN Question 2
# Build KNN model with 5 neighbors
knn = KNeighborsClassifier(n_neighbors=5)
# Fit using scaled training data
knn.fit(X_train_scaled, y_train)
# Predict on the scaled test data
pred = knn.predict(X_test_scaled)
# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Scaling did not improve performance because all features were already measured in similar units and had similar ranges,
# so it was not necessary for this dataset.
# Scaling can help when a dataset has features with very different ranges.

# KNN Question 3
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("Fold scores:")
for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.3f}")

print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

# This result is more trustworthy than a single train/test split because
# cross-validation tests the model on multiple different parts of the training data,
# giving a more reliable estimate of its performance.

# KNN Question 4
k_values = [1, 3, 5, 7, 9, 11, 13, 15]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
    print(f"K={k}, Mean CV Score: {cv_scores.mean():.3f}")
# I would choose k=5 because it had a high CV score and gives a good balance.
# Smaller k values may overfit, and larger k values may underfit.

# --- Classifier Evaluation ---
# Classifier Evaluation Question 1
# Create confusion matrix
cm = confusion_matrix(y_test, knn_unscaled_pred)
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(colorbar=False)
# Save the figure
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()
# The model most often confuses versicolor and virginica because they have similar features,
# making them harder to separate.

# --- The sklearn API: Decision Trees ---
# Decision Trees Question 1
# Create Decision Tree model
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
# Fit on unscaled training data
tree.fit(X_train, y_train)
# Predict on test data
tree_pred = tree.predict(X_test)
# Print accuracy and classification report
print("Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))
# Decision Trees and KNN can have similar accuracy.
# Decision Trees do not use distance calculations, 
# so scaling does not usually affect performance.

# --- Logistic Regression and Regularization ---
# Logistic Regression Question 1
c_values = [0.01, 1.0, 100]

for c in c_values:
    model = OneVsRestClassifier(LogisticRegression(C=c, max_iter=1000, solver='liblinear'))
    model.fit(X_train_scaled, y_train)
    # Sum coefficients from all OneVsRest Logistic Regression models
    coef_sum = sum(
        np.abs(estimator.coef_).sum()
        for estimator in model.estimators_
    )
    print(f"C={c}, Total Coefficient Magnitude={coef_sum:.3f}")
# As C increases, the total coefficient magnitude increases.
# This shows that larger C means weaker regularization, allowing larger coefficients.
# Smaller C means stronger regularization, which keeps the coefficients smaller.

# --- PCA ---
digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting
# PCA Question 1
print("X_digits Shape:", X_digits.shape)
print("Images Shape:", images.shape)
fig, ax = plt.subplots(1, 10, figsize=(12, 2)) 
for digit in range(10):
   idx = np.where(y_digits == digit)[0][0]
   ax[digit].imshow(images[idx],  cmap='gray_r')
   ax[digit].set_title(f"Digit {y_digits[idx]}")
   # Remove axis numbers
   ax[digit].axis('off')
plt.tight_layout()   
plt.savefig("assignments_03/outputs/sample_digits.png")
plt.close()
# PCA Question 2
pca = PCA()
pca.fit(X_digits)
scores = pca.transform(X_digits)
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.colorbar(scatter, label='Digit')
plt.title("PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.savefig("assignments_03/outputs/pca_2d_projection.png")
plt.close()
# Same-digit images mostly cluster together in this 2D space, 
# although some digits overlap because reducing 64 features to 2 loses some information.

# PCA Question 3
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.plot(cumulative_variance)
plt.axhline(y=0.80, color="red", linestyle="--", label="80%")
plt.legend()
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")

# Calculate how many components are needed to explain 80% of the variance
components_80 = np.argmax(cumulative_variance >= 0.80) + 1
print("Components needed for 80% variance:", components_80)

plt.savefig("assignments_03/outputs/pca_variance_explained.png")
plt.close()

# Approximately 13 components are needed to explain 80% of the variance.
# PCA reduces the number of features while keeping most of the important information.

# PCA Question 4
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

n_values = [2, 5, 15, 40]

# Use the first five samples from X_digits
sample_indices = [0, 1, 2, 3, 4]
fig, ax = plt.subplots(5, 5, figsize=(10, 10))
# Original images
for col, sample_idx in enumerate(sample_indices):
    ax[0, col].imshow(images[sample_idx], cmap="gray_r")
    ax[0, col].set_title(f"Digit {y_digits[sample_idx]}")
    ax[0, col].axis("off")
ax[0, 0].set_ylabel ("Original", fontsize=12)

# Reconstructed images
for row, n in enumerate(n_values, start=1):
    for col, sample_idx in enumerate(sample_indices):
        reconstructed = reconstruct_digit(sample_idx, scores, pca, n)
        ax[row, col].imshow(reconstructed, cmap="gray_r")
        ax[row, col].axis("off")
    ax[row, 0].set_ylabel(f"n={n}", fontsize=12)

plt.tight_layout()
plt.savefig("assignments_03/outputs/pca_reconstructions.png", bbox_inches="tight")
plt.close()
# Using more principal components makes the reconstructed digits look more like the originals.
# Around 15 components, the digits are clearly recognizable.