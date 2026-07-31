# --- ROC and AUC ---
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    RocCurveDisplay,
    classification_report,
)
import joblib

os.makedirs("assignments_04/outputs", exist_ok=True)
os.makedirs("assignments_04/models", exist_ok=True)

# Synthetic dataset — binary classification, two informative features
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=4,
    n_redundant=2,
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ROC Question 1
# Train a LogisticRegression(max_iter=1000, random_state=42) on the raw (unscaled) training data
log_reg_unscaled = LogisticRegression(max_iter=1000, random_state=42)
log_reg_unscaled.fit(X_train, y_train)

# Train KNeighborsClassifier(n_neighbors=5) on the scaled training data
# Fit the scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

# Compute predicted probabilities for Logistic Regression
log_reg_probs =log_reg_unscaled.predict_proba(X_test)[:, 1]

# Compute predicted probabilities for KNN
knn_probs = knn_scaled.predict_proba(X_test_scaled)[:, 1]

# Compute and print the AUC score
log_reg_auc = roc_auc_score(y_test, log_reg_probs)
knn_auc = roc_auc_score(y_test, knn_probs)
print("Logistic Regression AUC:", log_reg_auc)
print("KNN AUC:", knn_auc)

# KNN has a higher AUC than Logistic Regression.
# This means KNN separates the two classes better across all possible
# thresholds and has better overall classification performance.


# ROC Question 2
log_reg_fpr, log_reg_tpr, log_reg_thresholds = roc_curve(y_test, log_reg_probs)
knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_probs)

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay(fpr=log_reg_fpr, tpr=log_reg_tpr, roc_auc=log_reg_auc
                ).plot(ax=ax, name="Logistic Regression")

RocCurveDisplay(fpr=knn_fpr, tpr=knn_tpr, roc_auc=knn_auc
                ).plot(ax=ax, name="KNN")
ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
ax.set_title("ROC Curve Comparison")
ax.legend()
plt.tight_layout()
plt.savefig("assignments_04/outputs/roc_comparison.png")
plt.close()

# At TPR = 0.80, KNN has the lower FPR, 
# this means KNN catches 80% of positive cases with fewer false alarms.

# ROC Question 3
best_f1 = 0
best_threshold = None
best_tpr = None
best_fpr = None
for i, threshold in enumerate(log_reg_thresholds):
    y_pred = (log_reg_probs >= threshold).astype(int)
    f1 = f1_score(y_test, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold
        best_tpr = log_reg_tpr[i]
        best_fpr = log_reg_fpr[i]

print("Best Threshold:", best_threshold)
print("TPR:", best_tpr)
print("FPR:", best_fpr)
print("Best F1:", best_f1)

# The best threshold is lower than 0.5.
# A lower threshold is useful when it is more important to catch more positive cases,
# even if it produces more false alarms.

# --- GridSearchCV ---
# GridSearch Question 1
# Create a Pipeline 
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("logreg", LogisticRegression(max_iter=1000, random_state=42))
])

# Search different C values
param_grid = {"logreg__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

# Use GridSearchCV with 5-fold cross-validation and ROC AUC scoring
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring="roc_auc"
)

# Train and search for the best C
grid_search.fit(X_train, y_train)

print("Best C:", grid_search.best_params_["logreg__C"])
print("Best CV AUC:", grid_search.best_score_)

# Calculate the test AUC of the best estimator
best_pipe = grid_search.best_estimator_

y_pred  = best_pipe.predict(X_test)    # no manual scaling needed
y_probs = best_pipe.predict_proba(X_test)[:, 1]

print(f"Test AUC: {roc_auc_score(y_test, y_probs):.3f}")

# Grid search selected C=100.0 instead of the default C=1.0.
# The test AUC stayed the same (0.706), so the pipeline with scaling
# and a different C value did not improve the test performance.

# GridSearch Question 2
# Create a Pipeline 
tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("tree", DecisionTreeClassifier(random_state=42))
])

# Search different max_depth values
tree_param_grid = {
    "tree__max_depth": [2, 3, 5, 8, None]
}

# Use GridSearchCV with 5-fold cross-validation and ROC AUC scoring
tree_grid_search = GridSearchCV(
    tree_pipeline,
    tree_param_grid,
    cv=5,
    scoring="roc_auc"
)

# Train and search for the best max_depth
tree_grid_search.fit(X_train, y_train)

print("Best max_depth:", tree_grid_search.best_params_["tree__max_depth"])
print("Best CV AUC:", tree_grid_search.best_score_)

# Calculate the test AUC of the best decision tree
best_tree = tree_grid_search.best_estimator_

tree_probs = best_tree.predict_proba(X_test)[:, 1]
tree_test_auc = roc_auc_score(y_test, tree_probs)
print(f"Test AUC: {tree_test_auc:.3f}")

# The Decision Tree had a higher test AUC (0.935) than Logistic Regression (0.706),
# so I would choose the Decision Tree for further development.
# However, AUC is not the only thing to consider. I would also look at
# overfitting, how fast the model runs, and how easy it is to understand.

# GridSearch Question 3
results = tree_grid_search.cv_results_
sorted_results = sorted(
    zip(
        results["params"],
        results["mean_test_score"],
        results["std_test_score"] 
    ),
    key=lambda x: x[1],
    reverse=True
)
for params, mean_score, std_score in sorted_results:
    print(
        params,
        "Mean CV AUC:",
        round(mean_score, 3),
        "Std:",
        round(std_score, 3)
    )

# max_depth=5 and max_depth=3 have similar mean AUC scores.
# I would choose max_depth=3 because it has a slightly lower standard deviation,
# so its results are a little more consistent.

# --- joblib ---
# joblib Question 1
# Get the best logistic regression pipeline from GridSearch
best_lr_pipe = grid_search.best_estimator_

# Save the model 
joblib.dump(best_lr_pipe, "assignments_04/models/warmup_model.pkl")

# Load the model 
loaded_clf = joblib.load("assignments_04/models/warmup_model.pkl")

original_preds = best_lr_pipe.predict(X_test)
loaded_preds   = loaded_clf.predict(X_test)

assert (original_preds == loaded_preds).all(), "Predictions do not match!"
print("Predictions match. Model saved and loaded successfully.")

# If we saved only the Logistic Regression model, it would use unscaled data
# for prediction, which could give incorrect results. The pipeline saves
# both the scaler and the model, so the data is scaled before predicting.

# joblib Question 2
# --- Simulated prediction script ---
loaded_model = joblib.load("assignments_04/models/warmup_model.pkl")

# Three hand-crafted test cases — raw, unscaled data
new_samples = np.array([
    [2.5,  1.2, -0.3,  0.8,  1.0, -0.5,  0.2,  0.9, -1.1,  0.4],
    [-1.0, 0.5,  0.9, -0.7, -0.2,  1.3, -0.8,  0.1,  0.5, -0.3],
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0],
])

# Predict classes
predictions = loaded_model.predict(new_samples)

# Predict probabilities
probabilities = loaded_model.predict_proba(new_samples)

for i in range(len(new_samples)):
    print(f"Sample {i+1}")
    print("Predicted class:", predictions[i])
    print("Probability:", probabilities[i])

# The all-zeros row was predicted as class 1 because the model gave
# a higher probability to class 1 than class 0.