import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from prefect import task, flow, get_run_logger

# Task 1: Load Multiple Years of Data
@task (retries=3, retry_delay_seconds=2)
def load_and_merge_data():
    logger = get_run_logger()
    logger.info("Starting data loading")
    # To store all DataFrames
    dfs = []
    years = range(2015, 2025)
    for year in years:
        file_path = f"happiness_project/world_happiness_{year}.csv"
        # Files contains (; and , )
        df = pd.read_csv(file_path, sep=";", decimal=",")

        # Rename "Ladder score" to "Happiness score" column
        df.rename(columns={"Ladder score": "Happiness score"}, inplace=True)
        
        # Add year column 
        df["year"] = year

        dfs.append(df)

    # store all dfs 
    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv("outputs/merged_happiness.csv", index=False)
    logger.info("Saved merged_happiness.csv")
    logger.info(f"Columns: {merged_df.columns.tolist()}")
    return merged_df
    
# Task 2: Descriptive Statistics
@task
def descriptive_stats(df):
    logger = get_run_logger()
    logger.info("Starting descriptive statistics")

    # Calculate mean, median, std 
    mean_score = df["Happiness score"].mean()
    median_score = df["Happiness score"].median()
    std_score = df["Happiness score"].std()

    logger.info(f"Mean happiness score: {mean_score}")
    logger.info(f"Median happiness score: {median_score}")
    logger.info(f"Standard deviation score: {std_score}")

    # Mean by year
    mean_by_year = df.groupby("year")["Happiness score"].mean()
    logger.info(f"Average score per year:\n{mean_by_year}")
    # Mean by region
    mean_by_region = df.groupby("Regional indicator")["Happiness score"].mean()
    logger.info(f"Average score per region:\n{mean_by_region}")

    return {
        "mean": mean_score,
        "median": median_score,
        "std": std_score,
        "year_mean": mean_by_year,
        "region_mean": mean_by_region
    }

# Task 3: Visual Exploration
@task
def visual_exploration(df):
    logger = get_run_logger()
    logger.info("Starting Visual Exploration")

    # Create histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(df["Happiness score"], bins=20)
    plt.title("Distribution of Happiness Scores")
    plt.xlabel("Happiness Score")
    plt.ylabel("Count")
    plt.savefig("outputs/happiness_histogram.png")
    plt.close()

    logger.info("Saved happiness histogram")

    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="year", y="Happiness score")
    plt.title("Happiness Score by Year")
    plt.xlabel("Year")
    plt.ylabel("Happiness Score")
    plt.savefig("outputs/happiness_by_year.png")
    plt.close()

    logger.info("Saved happiness by year boxplot")

    # Create scatter plot
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="GDP per capita", y="Happiness score")
    plt.title("GDP per Capita vs Happiness Score")
    plt.xlabel("GDP per Capita")
    plt.ylabel("Happiness Score")
    plt.savefig("outputs/gdp_vs_happiness.png")
    plt.close()

    logger.info("Saved GDP vs happiness scatter plot")

    # Create heatmap
    plt.figure(figsize=(12, 8))
    numeric_df = df.select_dtypes(include="number")
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    logger.info("Saved correlation heatmap")

# Task 4: Hypothesis Testing
@task
def hypothesis_testing(df):
    logger = get_run_logger()
    logger.info("Starting Hypothesis Testing")
    scores_2019 = df[df["year"]==2019]["Happiness score"]
    scores_2020 = df[df["year"]==2020]["Happiness score"]
    t_stat, p_value = stats.ttest_ind(scores_2019, scores_2020)
    logger.info(f"2019 vs 2020 T-statistic: {t_stat}")
    logger.info(f"2019 vs 2020 P-value: {p_value}")


    # Calculate mean 
    mean_2019 = scores_2019.mean()
    mean_2020 = scores_2020.mean()
    logger.info(f"2019 Mean: {mean_2019}")
    logger.info(f"2020 Mean: {mean_2020}")
    
    if p_value < 0.05:
        logger.info("The difference in happiness scores between 2019 and 2020 is statistically significant.")
    else:
        logger.info("No statistically significant difference detected in happiness scores between 2019 and 2020.")
    
    # Region comparison
    western_europe = df[df["Regional indicator"] == "Western Europe"]["Happiness score"]
    middle_east_and_north_africa = df[df["Regional indicator"] == "Middle East and North Africa"]["Happiness score"]

    t_stat2, p_value2 = stats.ttest_ind(western_europe, middle_east_and_north_africa)
    logger.info(f"Western Europe vs Middle East and North Africa T-statistic: {t_stat2}")
    logger.info(f"Western Europe vs Middle East and North Africa P-value: {p_value2}")

    # Calculate mean 
    mean_western_europe = western_europe.mean()
    mean_middle_east_and_north_africa= middle_east_and_north_africa.mean()
    logger.info(f"Western Europe Mean: {mean_western_europe}")
    logger.info(f"Middle East and North Africa Mean: {mean_middle_east_and_north_africa}")

    if p_value2 < 0.05:
        logger.info("The difference in happiness scores between Western Europe and Middle East and North Africa is statistically significant")
    else:
        logger.info("No statistically significant difference detected between Western Europe and Middle East and North Africa")
    
    return {
        "p_value": p_value,
        "mean_2019": mean_2019,
        "mean_2020": mean_2020
    }

    
 # Task 5: Correlation and Multiple Comparisons
@task
def correlation_and_multiple_comparisons(df):
    logger = get_run_logger()
    logger.info("Starting Correlation and Multiple Comparisons")

    # Variables to compare with Happiness score
    variables = [
       "GDP per capita",
       "Social support",
       "Healthy life expectancy",
       "Freedom to make life choices",
       "Generosity",
       "Perceptions of corruption"
       ]
    # Count number of correlation tests
    number_of_tests = len(variables)
    logger.info(f"Number of correlation tests: {number_of_tests}")
    
    # Bonferroni correction
    adjusted_alpha = 0.05 / number_of_tests
    logger.info(f" Bonferroni adjusted alpha: {adjusted_alpha}")

    # Store results
    corr_results = {}

    # Calculate Pearson correlation for each variable
    for variable in variables:
        corr_coef, p_value = stats.pearsonr(df[variable], df["Happiness score"])
        corr_results[variable] = (corr_coef, p_value)

        logger.info(f"Variables: {variable}")
        logger.info(f"Correlation Coefficient: {corr_coef}")
        logger.info(f"P-value: {p_value}")
        
        # Original alpha
        if p_value < 0.05:
            logger.info(f"{variable} is significantly correlated with Happiness score")
        else:
            logger.info(f"{variable} is not significantly correlated with Happiness score")    
        
        # Bonferroni correction
        if p_value < adjusted_alpha:
            logger.info(f"{variable} remains significant after Bonferroni correction")
        else:
            logger.info(f"{variable} is not significant after Bonferroni correction")

    return corr_results

# Task 6: Summary Report
@task
def summary_report(df, corr_results, test_results):
    logger = get_run_logger()
    logger.info("Starting Summary Report")  

    # Total number of countries and years
    total_countries = df["Country"].nunique()
    total_years = df["year"].nunique()
    logger.info(f"Total number of countries: {total_countries}")
    logger.info(f"Total number of years: {total_years}")
    
    # The top 3 and bottom 3 regions
    region_mean = df.groupby("Regional indicator")["Happiness score"].mean()
    top_3_regions = region_mean.nlargest(3)
    bottom_3_regions = region_mean.nsmallest(3)
    logger.info(f"Top 3 happiest regions:\n{top_3_regions}")
    logger.info(f"Bottom 3 happiest regions:\n{bottom_3_regions}")
    
    # The result of the pre/post-2020 t-test
    if test_results["p_value"] < 0.05:
        logger.info(
            "The analysis shows a statistically significant difference "
            "between happiness scores in 2019 and 2020."
        )
    else:
        logger.info(
            "The analysis shows no statistically significant difference "
            "between happiness scores in 2019 and 2020."
        )
    
    # The variable most strongly correlated
    strongest_variable = max(corr_results, key=lambda x: abs(corr_results[x][0]))
    strongest_corr = corr_results[strongest_variable][0]
    
    logger.info(f"Variable most strongly correlated with happiness: {strongest_variable} {strongest_corr}") 


@flow
def happiness_pipeline():
    merged_df = load_and_merge_data()
    descriptive_stats(merged_df)
    visual_exploration(merged_df)
    test_results = hypothesis_testing(merged_df)
    corr_results = correlation_and_multiple_comparisons(merged_df)
    summary_report(merged_df, corr_results, test_results)
if __name__ == "__main__":
    happiness_pipeline()