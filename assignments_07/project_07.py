from smolagents import tool
import pandas as pd
from scipy.stats import pearsonr
import os

# Path to the merged Week 1 dataset
DATA_PATH = "assignments_01/outputs/merged_happiness.csv"

# ================================================================
# Task 1: Define Your Tools
# ================================================================

# Shared DataFrame 
df = None

@tool
def load_happiness_data() -> dict:
    """Load the World Happiness dataset into memory.

    Loads the merged World Happiness CSV from DATA_PATH. If the merged
    file does not exist, loads and merges the yearly World Happiness CSV
    files from the happiness project resources directory.

    Returns:
        dict: A dictionary containing the dataset shape and column names.
    """
    global df

    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
    else:
        dfs = []

        for year in range(2015, 2025):
            file_path = (
                f"assignments_01/happiness_project/"
                f"world_happiness_{year}.csv"
            )

            yearly_df = pd.read_csv(
                file_path,
                sep=";",
                decimal=","
            )

            yearly_df.rename(
                columns={"Ladder score": "Happiness score"},
                inplace=True
            )

            yearly_df["year"] = year
            dfs.append(yearly_df)

        df = pd.concat(dfs, ignore_index=True)

    return {
        "shape": df.shape,
        "columns": df.columns.tolist()
    }
@tool
def summarize_column(column: str) -> dict:
    """Return descriptive statistics for a single column 
    in the loaded dataset.

    Args:
      column: The name of column to summarize.


    Returns:
      dict: Descriptive statistics for the requested column, or an
      error message if the data is not loaded or the column does not exist.
    """
    if df is None:
        return {"error": "Data is not loaded."}
    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    try:
        return df[column].describe().to_dict()
    except Exception as e:
        return {"error": str(e)}
    
@tool
def compute_correlation(col1: str, col2: str) -> dict:
    """Compute the Pearson correlation coefficient and 
    p-value between two numeric columns.
    
    Args:
      col1: The name of the first numeric column.
      col2: The name of the second numeric column.
    Returns:
      dict: A dictionary containing the two column names, the Pearson
      correlation coefficient, and the p-value.
    """
    if df is None:
        return {"error": "Data is not loaded."}
    if col1 not in df.columns:
        return {"error": f"Column '{col1}' does not exist."}
    if col2 not in df.columns:
        return {"error": f"Column '{col2}' does not exist."}

    try:
        data = df[[col1, col2]].dropna()
        r, p_value = pearsonr(data[col1], data[col2])

        return {
            "col1": col1,
            "col2": col2,
            "pearson_r": round(r, 4),
            "p_value": round(p_value, 4)
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_top_n_countries(column: str, year: int, n: int = 5) -> dict:
    """Return the top N countries ranked by a given column 
    for a specific year.

    Args:
        column: The column used to rank the countries.
        year: The year to filter the dataset by.
        n: The number of countries to return.

      Returns:
        A list of dictionaries containing each country's name and
        value for the requested column, or an error dictionary if
        the data or requested inputs are invalid.
    """ 
    if df is None:
        return {"error": "Data is not loaded."}

    if column not in df.columns:
        return {"error": f"Column '{column}' does not exist."}

    try:
        year_data = df[df["year"] == year]
        
        top_n = (
            year_data
            .sort_values(column, ascending=False)
            .head(n)
        )

        return [
            {
                "country": row["Country"],
                column: row[column]
            }
            for _, row in top_n.iterrows()
        ]

    except Exception as e:
        return {"error": str(e)}   

# ================================================================
# Task 2: Build the Agent
# ================================================================

from smolagents import CodeAgent, OpenAIServerModel
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIServerModel(
    api_key=api_key,
    model_id="gpt-4o-mini"
)

SYSTEM_PROMPT = """
You are a data analyst assistant for the World Happiness dataset.
Use the available tools for loading data, summarizing columns, computing correlations,
and ranking countries. Write Python code directly only when the tools are not sufficient
(for example, when creating custom plots or computing something the tools don't cover).
Be concise and student-friendly in your responses.
"""

agent = CodeAgent(
    tools=[load_happiness_data, summarize_column, compute_correlation, get_top_n_countries],
    model=model,
    instructions=SYSTEM_PROMPT,
    additional_authorized_imports=["pandas", "matplotlib.pyplot", "scipy.stats"],
    max_steps=8,
) 

# ================================================================
# Task 3: Run Guided Queries
# ================================================================

queries = [
    "Load the happiness data and tell me its shape and column names.",
    "Summarize the Happiness score column.",
    "What is the correlation between GDP per capita and Happiness score? Is it statistically significant?",
    "Show me the top 5 happiest countries in 2020.",
    "Plot Happiness score over the years as a line chart, with one line per region using Regional indicator. Use pandas to read assignments_01/outputs/merged_happiness.csv directly. Save the plot to assignments_07/outputs/happiness_by_region.png.",
]

if __name__ == "__main__":

    for query in queries:
        print(f"\n--- Query: {query} ---")
        response = agent.run(query, reset=False)
        print(response)

    # ================================================================
    # Task 4: Your Own Questions
    # ================================================================

    # My query 1
    my_query_1 = "What are the descriptive statistics for Happiness score?"  
    response_1 = agent.run(my_query_1, reset=False)
    print(response_1)
    # # Comment: This triggered tool use because the agent used the summarize_column tool.

    # My query 2
    my_query_2 = """
    Create a histogram of the Happiness score using the actual dataset.
    Use pandas to read assignments_01/outputs/merged_happiness.csv directly,
    then use matplotlib to create the histogram.
    Do not use mock or simulated data.
    Save the plot to assignments_07/outputs/happiness_histogram.png.
    """

    response_2 = agent.run(my_query_2, reset=False)
    print(response_2)
    # Comment: This triggered code generation because the agent wrote
    # pandas and matplotlib code to create and save the histogram.

# ================================================================
# Task 5: Reflection
# ================================================================

# 1. In Query 3, how did the agent communicate whether the correlation was
# statistically significant? Did it use the p-value correctly? What
# threshold did it apply?

# The agent reported a Pearson correlation of 0.6313 and a p-value of 0.0.
# Since the p-value is below 0.05, the correlation is statistically significant.


# 2. Did any of the agent's responses surprise you — either by being more
# capable than you expected, or less? Describe one specific example.

# I was surprised that the agent could write pandas and matplotlib code
# by itself to create charts. I also noticed that it sometimes needed
# several steps to handle errors when working with the data and plots.


# 3. What one additional tool would make this agent meaningfully more useful?

# I would add a tool to compare happiness scores for countries or regions
# across different years. This would make it easier to analyze changes
# over time.