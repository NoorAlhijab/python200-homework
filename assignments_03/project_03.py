import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from io import BytesIO

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

# Task 1: Load and Explore

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
print(df.shape)
print(df.head())
# Check how many spam/ham values 
print(df["spam_label"].value_counts())
# The dataset is fairly balanced.
# Ham emails: 2788 (about 61%)
# Spam emails: 1813 (about 39%)
# Accuracy will be usel metric.

# Boxplot for key features
features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]
for feature in features:
    plt.figure(figsize=(6, 4))

    df.boxplot(column=feature, by="spam_label")
    plt.title(feature)
    plt.xlabel("Spam Label (0 = Ham, 1 = Spam)")
    plt.ylabel(feature)
    plt.savefig(f"assignments_03/outputs/{feature}_boxplot.png")
    plt.close()

# Observations from feature distributions:
#
# word_freq_free:
# Spam emails usually contain more occurrences of the word "free" because
# many spam messages use promotional language and offers.
#
# char_freq_!:
# Spam emails tend to have higher exclamation mark frequency because
# advertisements and spam messages often use urgent or attention-grabbing text.
#
# capital_run_length_total:
# Spam emails often contain longer sequences of capital letters because
# they may use uppercase words to emphasize deals, warnings, or promotions.



# Task 2: Prepare Your Data

# Separating Features and Target
X = df.drop("spam_label", axis=1)
y = df["spam_label"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA()
pca.fit(X_train_scaled)

# cumulative explained variance
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8, 5))
plt.plot(cumulative_variance)
plt.title("PCA cumulative explained variance")
plt.xlabel("Number of Components")
plt.ylabel("cumulative explained variance")
plt.savefig("assignments_03/outputs/pca_cumulative_explained_variance.png")
plt.close()

# Find n components reaching 90%
components_90 = np.argmax(cumulative_variance >= 0.90) + 1
print("Components needed for 90% variance:", components_90)

# Transform data
X_train_pca = pca.transform(X_train_scaled)[:, :components_90]
X_test_pca  = pca.transform(X_test_scaled)[:, :components_90]

# Task 3: A Classifier Comparison

# Build KNN  unsclaed model with 5 neighbors
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
# Fit using unscaled training data
knn_unscaled.fit(X_train, y_train)
# Predict on the test data
pred = knn_unscaled.predict(X_test)
# Print accuracy and classification report
print("KNN Unscaled Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Build KNN scaled model with 5 neighbors 
knn_scaled = KNeighborsClassifier(n_neighbors=5)
# Fit using scaled training data
knn_scaled.fit(X_train_scaled, y_train)
# Predict on the test data
pred = knn_scaled.predict(X_test_scaled)
# Print accuracy and classification report
print("KNN Scaled Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Unscaled accuracy is about 0.89 and scaled accuracy is about 0.90, 
# that means KNN preformed better with scaled data

# KNN PCA
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
pred = knn_pca.predict(X_test_pca)
print("KNN PCA Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Decision Tree
max_depth_values =[3, 5, 10, None]
for depth in max_depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    print("Depth:", depth)
    # Print the training accuracy and the test accuracy
    print("Train Accuracy:", tree.score(X_train, y_train))
    print("Test Accuracy:", tree.score(X_test, y_test))

# Choose depth 
tree_final = DecisionTreeClassifier(max_depth=10, random_state=42)
tree_final.fit(X_train, y_train)
tree_pred = tree_final.predict(X_test)
print("Tree Accuracy:", accuracy_score(y_test, tree_pred))
print(classification_report(y_test, tree_pred))

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# Feature importance
tree_importance = pd.Series(
    tree_final.feature_importances_,
    index=X.columns
)

print("Decision Tree Top Features")
print(tree_importance.sort_values(ascending=False).head(10))


rf_importance = pd.Series(
    rf.feature_importances_,
    index=X.columns
)

print("Random Forest Top Features")
print(rf_importance.sort_values(ascending=False).head(10))

plt.figure(figsize=(10,6))

rf_importance.sort_values(ascending=False).head(10).plot(
    kind="bar"
)

plt.title("Top 10 Random Forest Feature Importances")
plt.ylabel("Importance")

plt.savefig(
    "assignments_03/outputs/feature_importances.png"
)

plt.close()

# Logistic Regression Scaled
log_reg_scaled = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_scaled.fit(X_train_scaled, y_train)
log_pred_scaled = log_reg_scaled.predict(X_test_scaled)

print("Logistic Regression Scaled Accuracy:", accuracy_score(y_test, log_pred_scaled))
print(classification_report(y_test, log_pred_scaled))

# Logistic Regression PCA
log_reg_pca = LogisticRegression(C=1.0, max_iter=1000, solver='liblinear')
log_reg_pca.fit(X_train_pca, y_train)
log_pred_pca = log_reg_pca.predict(X_test_pca)

print("Logistic Regression PCA Accuracy:", accuracy_score(y_test, log_pred_pca))
print(classification_report(y_test, log_pred_pca))

# Model comparison:
#
# Random Forest performed the best based on accuracy because tree-based
# models can learn complex patterns without requiring feature scaling.
#
# KNN comparison:
# The scaled KNN model performed better than the unscaled version because
# KNN uses distance calculations, and scaling prevents larger-range features
# from dominating the distance.
#
# The PCA version of KNN reduced the number of features while keeping most
# important information. However, the improvement was small compared with
# the scaled version.
#
# Logistic Regression comparison:
# The scaled Logistic Regression model performed better than the PCA version.
# This shows that removing features with PCA did not improve performance
# for Logistic Regression on this dataset.
#
# For spam detection, accuracy alone is not enough. False positives are
# important because legitimate emails could incorrectly be marked as spam.
# Therefore, precision and recall should also be considered.

# Confusion matrix for best model
disp = ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.title("Best Model Confusion Matrix")
plt.savefig("assignments_03/outputs/best_model_confusion_matrix.png")
plt.close()

# The confusion matrix shows the number of false positives and false negatives.
#
# False positives are legitimate emails incorrectly classified as spam.
# False negatives are spam emails incorrectly classified as ham.
#
# For spam filtering, false positives are usually the more concerning error
# because an important email could be incorrectly moved to the spam folder.
# False negatives are less harmful because the user can delete an unwanted email.

# Task 4: Cross-Validation
# KNN Unscaled
knn_unscaled_cv = cross_val_score(knn_unscaled, X_train, y_train, cv=5)
print("KNN Unscaled")
print("Mean:", knn_unscaled_cv.mean())
print("std:", knn_unscaled_cv.std())

# KNN Scaled
knn_scaled_cv = cross_val_score(knn_scaled, X_train_scaled, y_train, cv=5)
print("KNN Scaled")
print("Mean:", knn_scaled_cv.mean())
print("std:", knn_scaled_cv.std())

# KNN PCA
knn_pca_cv = cross_val_score(knn_pca, X_train_pca, y_train, cv=5)
print("KNN PCA")
print("Mean:", knn_pca_cv.mean())
print("std:", knn_pca_cv.std())

# Decision Tree
tree_cv = cross_val_score(tree_final, X_train, y_train, cv=5)
print("Decision Tree")
print("Mean:", tree_cv.mean())
print("std:", tree_cv.std())

# Random Forest
rf_cv = cross_val_score(rf, X_train, y_train, cv=5)
print("Random Forset")
print("Mean:", rf_cv.mean())
print("std:", rf_cv.std())

# Logistic Regression Scaled
log_reg_scaled_cv = cross_val_score(log_reg_scaled, X_train_scaled, y_train, cv=5)
print("Logistic Regression Scaled")
print("Mean:", log_reg_scaled_cv.mean())
print("std:", log_reg_scaled_cv.std())

# Logistic Regression PCA
log_reg_pca_cv = cross_val_score(log_reg_pca, X_train_pca, y_train, cv=5)
print("Logistic Regression PCA")
print("Mean:", log_reg_pca_cv.mean())
print("std:", log_reg_pca_cv.std())

# Cross-validation summary:
#
# Random Forest had the highest average accuracy across folds,
# showing the strongest overall performance.
#
# Logistic Regression PCA had the lowest standard deviation,
# meaning it produced the most consistent results across different folds.
#
# The cross-validation results are similar to the single train/test split,
# which increases confidence that the model comparison is reliable.

# Task 5: Building a Prediction Pipeline

# Random Forest pipeline 
rf_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
rf_pipeline.fit(X_train, y_train)
rf_pred =rf_pipeline.predict(X_test)
print("Random Forest Pipeline")
print(classification_report(y_test, rf_pred))

# Logistic Regression Pipline
log_pipeline = Pipeline([
    ("scaler", StandardScaler()), 
    ("classifier", LogisticRegression(
        C=1.0, 
        max_iter=1000, 
        solver='liblinear'
        ))
    ])
log_pipeline.fit(X_train, y_train)
log_pred = log_pipeline.predict(X_test)
print("Logistic Regression Pipeline")
print(classification_report(y_test, log_pred))

# Pipeline comparison:
#
# The Random Forest pipeline produces the same results as the manual Random
# Forest model because the pipeline only contains the classifier and does not
# apply any preprocessing. Tree-based models do not require feature scaling.
#
# The Logistic Regression pipeline includes StandardScaler because logistic
# regression performs better when features are on similar scales.
#
# The pipeline approach creates a reusable workflow by combining preprocessing
# and modeling steps together and helps prevent data leakage.



