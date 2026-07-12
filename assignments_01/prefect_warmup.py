# --- Pipelines  ---
# Pipeline Question 2
import pandas as pd
import numpy as np 
from prefect import task, flow

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

#  Create a pandas Series 
@task
def create_series(arr):
    return pd.Series(arr, name="values")

# Removes any NaN values
@task
def clean_data(series):
    return series.dropna()

# Calculate summary statistics
@task
def summarize_data(series):
    return {
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "mode": series.mode()[0]
    }

# Create Prefect flow
@flow
def pipeline_flow(arr):
    series = create_series(arr)
    cleaned_data = clean_data(series)
    summary = summarize_data(cleaned_data)
    return summary

# Call the pipeline
if __name__ == "__main__":
    result = pipeline_flow(arr)

    for key, value in result.items():
        print(f"{key}: {value}")



# 1. This pipeline is simple -- just three small functions on a handful of numbers. Why might Prefect be more overhead than it is worth here?
# Prefect is not necessary for a simple pipeline because it is simple and we can do all analysis steps using normal Python functions.

# 2. Describe some realistic scenarios where a framework like Prefect could still be useful, even if the pipeline logic itself stays simple like in this case.
# If raw data is updated every day, for example at midnight, Prefect can handle the tasks automatically, schedule the workflow, and help reduce errors.