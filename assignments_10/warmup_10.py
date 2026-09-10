# --- ML vs. LLM in Pipelines ---
# ML/LLM Question 1

# The ML classifier produces a binary prediction, such as "good" or "skip".
# It is trained to classify data based on patterns it learned from training data.

# The LLM produces a recommendation or explanation in natural language.
# It is used to generate useful text based on the information provided to it.

# Each tool is used for a different purpose. The ML model is better for making
# a consistent classification, while the LLM is better for generating and
# explaining recommendations in natural language.

# If we used the LLM to make the binary good/skip prediction, the result could
# be less consistent because LLM responses can vary. If we used the ML model
# to write the recommendation, it would not be suitable because a classifier
# produces predictions, not natural-language recommendations.

# ML/LLM Question 2

# Converting a date string like "2023-07-04" to day-of-week
# Use deterministic code because converting a date to a day is a fixed calculation.
#
# Classifying a job posting as "entry-level", "mid-level", or "senior" based on freeform text
# Use an LLM because it can understand freeform job posting text.
#
# Predicting customer churn given 15 numeric features and a labeled training dataset
# Use a trained ML model because it can learn from the labeled data to predict customer churn.
#
# Normalizing inconsistent city names ("NYC", "New York City", "New York, NY") to a canonical form
# Use deterministic code because known city names can be mapped to one standard name.
#
# Summing a column of revenue figures
# Use deterministic code because adding revenue numbers is a simple calculation.

# ML/LLM Question 3

# Incremental processing means processing only new records instead of processing all records again.
# It is important because it saves time and cost and prevents duplicate or incorrect results.
# If the script re-processed all 365 records every time, it would use more resources and could create
# duplicate data or overwrite existing results unnecessarily.

# --- Prompt Design ---
# Prompt Question 1

# Alternative system prompt:
# "Write exactly two sentences. The first sentence must state the prediction.
# The second sentence must explain the reasoning. Be direct and practical."
#
# Validation:
# The validation logic should check that the response contains exactly two sentences
# instead of exactly one sentence.

# Prompt Question 2

import time
def call_with_retry(client, messages, max_retries=3):
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                return None

# I would use this in a production pipeline when an API call may fail temporarily,
# such as because of a network problem, timeout, or temporary service error.
# Retrying after a short delay can give the API another chance to complete the request
# instead of failing the entire pipeline immediately.