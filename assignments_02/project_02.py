import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# The csv file using a semicolon as a separator instead of a comma, so we need to specify that in the read_csv function. 
# Task 1: Load and Explore
df = pd.read_csv("assignments_02/outputs/student_performance_math.csv", sep=";")
# Print the shape, the first five rows, and the data types of all columns
print("Shape of the DataFrame:", df.shape)
print("First five rows of the DataFrame:\n", df.head())
print("Data types of all columns:\n", df.dtypes)

# Create plot a histogram of G3 with 21 bins
plt.hist(df["G3"], bins=21)
plt.title("Distribution of Final Math Grades")
plt.xlabel("G3")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.close()

# Task 2: Preprocess the Data
# Filter the G3=0 rows since it represent the students who didn't take the final exam
df_cleaned = df[df["G3"] != 0].copy()

# Print the shape before and after filtering
print("Shape of the original DataFrame:", df.shape)
print("Shape of the cleaned DataFrame:", df_cleaned.shape)

# Convert the yes/no columns to 1/0 
yes_no_columns = ["schoolsup", "internet", "higher", "activities"]
for col in yes_no_columns:
    df_cleaned[col] = df_cleaned[col].map({"yes": 1, "no": 0})

# Convert the sex column to 0/1
df_cleaned["sex"] = df_cleaned["sex"].map({"M": 1, "F": 0})

# Compute the Pearson correlation between absences and G3 on both the original dataset and the filtered one
corr_original = df["absences"].corr(df["G3"])
corr_cleaned = df_cleaned["absences"].corr(df_cleaned["G3"])
print("Correlation between absences and G3 in the original dataset:", corr_original)
print("Correlation between absences and G3 in the cleaned dataset:", corr_cleaned)
# In the original dataset, the correlation is close to zero because it includes
# students with G3 = 0, who were absent from the final exam rather than failing it.
# These rows weaken the relationship between absences and G3.
#
# After filtering out those rows, the correlation becomes negative, showing that
# among students who actually took the exam, higher absences are associated with
# lower final grades.

# Task 3: Exploratory Data Analysis
# Compute the Pearson correlation between each numeric feature and G3 on the filtered dataset
feature_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime",
    "failures", "absences", "freetime", "goout", "Walc",
    "schoolsup", "internet", "higher", "activities", "sex"
]

correlations = df_cleaned[feature_cols].corrwith(df_cleaned["G3"]).sort_values()
print("Pearson correlation between each feature and G3:\n", correlations)
# Failures has the strongest negative relationship with G3, showing that
# students with more past failures tend to have lower final grades.
# Medu (mother's education level) has the strongest positive relationship,
# followed by Fedu (father's education level) and studytime.

# Visualization: 
# Create a bar chart of the correlations
plt.figure(figsize=(10, 6)) 
correlations.plot(kind="bar", color='skyblue', edgecolor='black')
plt.title("Correlation Between Features and G3")
plt.xlabel("Features")
plt.ylabel("Correlation")
plt.tight_layout()
plt.savefig("assignments_02/outputs/correlations.png")
plt.close()
# This bar chart shows the correlation between each feature and G3.
# Failures has the strongest negative relationship, while Medu (mother's
# education level) has the strongest positive relationship.

# Create a scatter plot of absences vs G3
plt.figure(figsize=(8, 5))
plt.scatter(df_cleaned["absences"], df_cleaned["G3"], color='blue', alpha=0.3, s=50)
plt.title("Absences vs Final Math Grades (G3)") 
plt.xlabel("Absences")
plt.ylabel("G3")
plt.tight_layout()
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.close()
# This scatter plot shows the relationship between absences and final math grades (G3).
# It illustrates the negative correlation, where higher absences are associated with lower grades.

# Task 4: Baseline Model
# Create a base model to predict G3 using failures alone 
X_base = df_cleaned["failures"].values
y = df_cleaned["G3"].values
X_base = X_base.reshape(-1, 1)
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_base, y, test_size=0.2, random_state=42)

base_model =  LinearRegression()
base_model.fit(X_train_b, y_train_b)
y_pred = base_model.predict(X_test_b)

# Print the slope, RMSE, and R² on the test set
print("Slope:", base_model.coef_[0])
# RMSE
rmse = np.sqrt(np.mean((y_pred - y_test_b) ** 2))
print("RMSE:", rmse)
# R2
r2 = base_model.score(X_test_b, y_test_b)
print("R2:", r2)
# The slope shows that each additional failure lowers the predicted grade by
# about 1.43 points. The RMSE of 2.96 means predictions are typically off by
# about 3 points on a 0-20 scale. The R² of 0.09 is low, but expected because
# failures alone cannot explain most differences in student grades.

# Task 5: Build the Full Model
feature_cols = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc", "schoolsup",
    "internet", "higher", "activities", "sex"
]

X = df_cleaned[feature_cols].values
y = df_cleaned["G3"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# print train R² and test R²
train_r2 = model.score(X_train, y_train)
print("Train R2:", train_r2)
test_r2 = model.score(X_test, y_test)
print("Test R2:", test_r2)
# RMSE
rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
print("RMSE:", rmse)

# Feature coefficients
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# The full model performs better than the baseline model. Test R² increased
# from 0.089 to 0.263, and RMSE decreased from 2.96 to 2.66. This shows that
# using additional student features improves prediction compared to using
# failures alone.

# Train and test R² values are close, suggesting that the model is not
# overfitting and performs similarly on new data.

# The negative coefficient for schoolsup was surprising. This may be because
# students receiving extra support were already struggling academically,
# so the negative value does not mean support causes lower grades.

# I would keep features with stronger effects, such as failures, studytime,
# absences, Medu, and Fedu because they provide useful information for
# predicting grades. I would consider dropping features with very small
# coefficients, such as freetime and activities, because they add less value
# to the model.

# Task 6: Evaluate and Summarize
plt.figure(figsize=(8, 6))

# Scatter plot of predicted vs actual values
plt.scatter(y_pred, y_test, alpha=0.6, color="blue")

# Diagonal reference line (predicted = actual)
plt.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], color="red")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.tight_layout()
plt.savefig("assignments_02/outputs/predicted_vs_actual.png")
plt.close()

# The model predicts most grades close to the middle.
# It predicts some low grades too high and some high grades too low.

# Points above the line mean the model predicted too low.
# Points below the line mean the model predicted too high.

# Dataset size
print("Filtered dataset size:", len(df_cleaned))
print("Test set size:", len(y_test))
# The filtered dataset has 357 students, and the test set has 72 students.

# The model's predictions are usually about 2.7 grade points away from
# the actual grade (RMSE = 2.66).

# The model explains about 26% of the differences in final grades (R² = 0.26).

# Internet has the largest positive coefficient, while schoolsup has the
# largest negative coefficient.

# One surprising result was the negative coefficient for schoolsup.
# This may be because students receiving extra support were already
# struggling in school.

# Neglected Feature: The Power of G1
feature_cols_with_g1 = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures",
    "absences", "freetime", "goout", "Walc", "schoolsup",
    "internet", "higher", "activities", "sex", "G1"
]

X_g1 = df_cleaned[feature_cols_with_g1].values
y = df_cleaned["G3"].values

X_train_g1, X_test_g1, y_train_g1, y_test_g1 = train_test_split(X_g1, y, test_size=0.2, random_state=42)

# Create and train the model
model_g1 = LinearRegression()
model_g1.fit(X_train_g1, y_train_g1)

test_r2_g1 = model_g1.score(X_test_g1, y_test_g1)
print("Test R² with G1:", test_r2_g1)

# Adding G1 greatly improves the model because it is a strong predictor of G3.

# A high R² does not mean G1 causes G3. It only means G1 helps predict G3.

# This model is useful for finding students who may struggle after the first grading period.

# If teachers want to help students before G1 is available, they need to use
# other features such as studytime, failures, absences, and family background.
