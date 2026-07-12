# --- Pandas ---

# Pandas Q1
import pandas as pd

data = {
    "name":   ["Alice", "Bob", "Carol", "David", "Eve"],
    "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)
# Print the first three rows
print(f"First three rows:\n {df.head(3)}")

# Print the shape
print(f"Shape: {df.shape}")

# Print the data types of each column
print(f"Data types:\n {df.dtypes}")

# Pandas Q2
# Filter students who passed and have a grade above 80.
filtered_students = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"Students who passed and have a grade above 80:\n {filtered_students}")

# Pandas Q3
df["grade_curved"] = df["grade"] + 5
print(f"Updated DataFrame: \n {df}")

# Pandas Q4
df["name_upper"] = df["name"].str.upper()
print(f"name and name_upper columns:\n  {df[['name', 'name_upper']]}")

# Pandas Q5
# Group the DataFrame by "city" and compute the mean grade for each city
grouped_cities = df.groupby('city')['grade'].mean()
print(f"Mean grade by city:\n {grouped_cities}")

# Pandas Q6
# Replace the value "Austin" in the "city" column with "Houston"
df["city"] = df["city"].replace("Austin", "Houston")
print(f"name and city columns:\n {df[['name', 'city']]}")

# Pandas Q7
# Sort the DataFrame by "grade" in descending order and print the top 3 rows
df_sorted = df.sort_values(by="grade", ascending=False)
print(f"Grade sorted in descending order:\n {df_sorted.head(3)}")

# --- NumPy ---
import numpy as np
# NumPy Q1
# Create a 1D NumPy array from the list [10, 20, 30, 40, 50]. Print its shape, dtype, and ndim.
arr = np.array([10, 20, 30, 40, 50])
print(f"Shape: {arr.shape}")
print(f"Data type: {arr.dtype}")
print(f"Number of dimensions: {arr.ndim}")

# NumPy Q2
# Create the following 2D array and print its shape and size (total number of elements)
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"Shape: {arr.shape}")
print(f"Total number of elements: {arr.size}")

# NumPy Q3
# Using the 2D array from Q2, slice out the top-left 2x2 block and print it. The expected result is [[1, 2], [4, 5]]
slice_arr = arr[0:2, 0:2]
print(f"top-left 2x2 block:\n {slice_arr}")

# NumPy Q4
# Create a 3x4 array of zeros using a built-in command. Then create a 2x5 array of ones using a built-in command. Print both
zeros_arr = np.zeros((3, 4))
ones_arr = np.ones((2, 5))
print(f"Zeros array:\n {zeros_arr}")
print(f"Ones array:\n {ones_arr}")

# NumPy Q5
# Create an array using np.arange(0, 50, 5), print the array, its shape, mean, sum, and standard deviation
arr = np.arange(0, 50, 5)
print(f"Array:\n {arr}")
print(f"Shape: {arr.shape}")
print(f"Mean: {arr.mean()}")
print(f"Sum: {arr.sum()}")
print(f"Standard deviation: {arr.std()}")

# NumPy Q6
# Generate an array of 200 random values drawn from a normal distribution with mean 0 and standard deviation 1 
# (use np.random.normal()). Print the mean and standard deviation of the result.
random_arr = np.random.normal(loc=0, scale=1, size=200)
print(f"Mean: {random_arr.mean()}")
print(f"Standard deviation: {random_arr.std()}")

# --- Matplotlib ---

import matplotlib.pyplot as plt
# Matplotlib Q1
# Plot the following data as a line plot. Add a title "Squares", x-axis label "x", and y-axis label "y"

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

# Create the line plot
plt.plot(x, y)
plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Matplotlib Q2
# Create a bar plot for the following subject scores. Add a title "Subject Scores" and label both axes.

subjects = ["Math", "Science", "English", "History"]
scores   = [88, 92, 75, 83]

# Create bar plot
plt.bar(subjects, scores)
plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

# Matplotlib Q3
# Plot the two datasets below as a scatter plot on the same figure. Use different colors for each, add a legend, and label both axes.

x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.scatter(x1, y1, color="green", label="Dataset 1")
plt.scatter(x2, y2, color="orange", label="Dataset 2")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

# Matplotlib Q4
# Use plt.subplots() to create a figure with 1 row and 2 subplots side by side. In the left subplot, plot x vs y from Q1 as a line. 
# In the right subplot, plot the subjects and scores from Q2 as a bar plot. Add a title to each subplot and call plt.tight_layout() before showing
fig, axes = plt.subplots(1, 2)
# Left side
axes[0].plot(x, y)
axes[0].set_title("Squares")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
# Right side
axes[1].bar(subjects, scores)
axes[1].set_title("Subject Scores")
axes[1].set_xlabel("Subjects")
axes[1].set_ylabel("Scores")

plt.tight_layout()
plt.show()


# --- Descriptive Statistics ---

# Descriptive Stats Question 1
# Use NumPy to compute and print the mean, median, variance, and standard deviation

data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]
print(f"Mean: {np.mean(data)}")
print(f"Median: {np.median(data)}")
print(f"Variance: {np.var(data)}")
print(f"Standard Deviation: {np.std(data)}")

# Descriptive Stats Question 2
# Generate 500 random values from a normal distribution with mean 65 and standard deviation 10 (use np.random.normal(65, 10, 500))
# Plot a histogram with 20 bins. Add a title "Distribution of Scores" and label both axes.
random_values = np.random.normal(65, 10, 500)

# Creat a histogram 
plt.hist(random_values, bins=20, color="purple", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Score Range")
plt.ylabel("Number of Scores")
plt.show()

# Descriptive Stats Question 3
# Create a boxplot comparing the two groups below. Label each box ("Group A" and "Group B") and add a title "Score Comparison".
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]
# Hint: pass labels=["Group A", "Group B"] to plt.boxplot()
# Create boxplot 
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Scores")
plt.show()

# Descriptive Stats Question 4
# You are given two datasets: one normally distributed and one 'exponential' distribution.

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)
# Create side-by-side boxplots comparing the two distributions. Label each boxplot appropriately ("Normal" and "Exponential") and add a title "Distribution Comparison"
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Values")
plt.show()

# Then, add a comment in your code briefly noting which distribution is more skewed, 
# and which descriptive statistic (mean or median) would provide a more appropriate measure of central tendency for each distribution

# --- Comment ---

# The normal data shows fewer outlier values or big numbers, while the skewed data shows more big values and outliers, so the exponential distribution is more skewed.
# The mean is a better descriptive statistic for the normal data because the values are close together, while the median is better for the skewed data because it has outliers or large numbers.

# Descriptive Stats Question 5
# Print the mean, median, and mode of the following:
import statistics as stats
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

print(f"Mean: {np.mean(data1)}")
print(f"Median: {np.median(data1)}")
print(f"Mode: {stats.mode(data1)}")

print(f"Mean: {np.mean(data2)}")
print(f"Median: {np.median(data2)}")
print(f"Mode: {stats.mode(data2)}")

# Why are the median and mean so different for data2? Add your answer as a comment in the code.
# Because data2 has a large value (150) that affects the mean, so the mean and median are not close values.

# --- Hypothesis  ---

# Hypothesis Question 1
# Run an independent samples t-test on the two groups below. Print the t-statistic and p-value.

from scipy import stats

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]
t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Question 2
# Using the p-value from Q1, write an if/else statement that prints whether the result is statistically significant at alpha = 0.05

if p_value < 0.05:
    print("The difference is statistically significant")
else:
    print("No statistically significant difference detected")

# Hypothesis Question 3
# Run a paired t-test on the before/after scores below (the same students measured twice). Print the t-statistic and p-value

before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_value = stats.ttest_rel(before, after)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Question 4
# Run a one-sample t-test to check whether the mean of scores is significantly different from a national benchmark of 70. Print the t-statistic and p-value
scores = [72, 68, 75, 70, 69, 74, 71, 73]
t_stat, p_value = stats.ttest_1samp(scores, 70)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")

# Hypothesis Question 5
# Re-run the test from Q1 as a one-tailed test to check whether group_a scores are less than group_b scores. Print the resulting p-value. Use the alternative parameter
t_stat, p_value = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"p-value: {p_value}")

# Hypothesis Question 6
# Write a plain-language conclusion for the result of Q1 (do not just say "reject the null hypothesis")
# Format it as a print() statement. Your conclusion should mention the direction of the difference and whether it is likely due to chance.
print("Group B scored higher than Group A, and this difference is unlikely to be due to chance.")

# --- Correlation  ---

# Correlation Question 1
# Compute the Pearson correlation between x and y below using np.corrcoef(). 
# Print the full correlation matrix, then print just the correlation coefficient (the value at position [0, 1])
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

corr_matrix = np.corrcoef(x, y)
print(f"Correlation matrix:\n {corr_matrix}")
print(f"Correlation coefficient: {corr_matrix[0, 1]}")

# What do you expect the correlation to be, and why? Add your answer as a comment in the code
# The relation is positive between X and y since x and y increase in same pattren that means correction is 1.

# Correlation Question 2
# Use pearsonr() from scipy.stats to compute the correlation between x and y below. Print both the correlation coefficient and the p-value
from scipy.stats import pearsonr

x = [1,  2,  3,  4,  5,  6,  7,  8,  9, 10]
y = [10, 9,  7,  8,  6,  5,  3,  4,  2,  1]

corr_coef, p_value = pearsonr(x, y)
print(f"Correlation: {corr_coef}")
print(f"p-value: {p_value}")

# Correlation Question 3
# Create the following DataFrame and use df.corr() to compute the correlation matrix. Print the result.
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)

corr_matrix = df.corr()
print(f"Correlation matrix:\n {corr_matrix}")

# Correlation Question 4
# Create a scatter plot of x and y below, which have a negative relationship. Add a title "Negative Correlation" and label both axes.
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]
plt.scatter(x, y)
plt.title("Negative Correlation")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Correlation Question 5
# Using the correlation matrix from Q3, create a heatmap with sns.heatmap(). 
# Pass annot=True so the correlation values appear in each cell, and add a title "Correlation Heatmap"

import seaborn as sns

sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# --- Pipelines  ---

# Pipeline Question 1
arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

#  Create a pandas Series 
def create_series(arr):
    return pd.Series(arr, name="values")

# Removes any NaN values
def clean_data(series):
    return series.dropna()

# Calculate summary statistics
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Calls the three functions
def data_pipeline(arr):
    series = create_series(arr)
    cleaned_data = clean_data(series)
    summary = summarize_data(cleaned_data)
    return summary

# Call the pipeline
result = data_pipeline(arr)

for key, value in result.items():
    print(f"{key}: {value}")