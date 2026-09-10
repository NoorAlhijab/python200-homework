# Video link: https://youtu.be/YISUyeEnM6M
import os
import json
import pandas as pd
import joblib
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI


def get_client():
    load_dotenv()
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url:
        raise ValueError("Supabase URL is missing.")
    if not key:
        raise ValueError("Supabase key is missing.")
    
    return create_client(url, key)

# =========================
# Step 1: Incremental Read
# =========================

def incremental_read(supabase):
    # Load model metadata from a JSON file
    with open("models/weather_classifier_metadata.json", "r") as f:
        metadata = json.load(f)

    # Get the weather records
    raw_rows = (supabase.table("weather_raw").select("*").execute().data)

    # Get dates that have already been enriched
    enriched_rows = (supabase.table("weather_enriched").select("date").execute().data)

    already_done = {row["date"] for row in enriched_rows}

    # Keep only records that have not been enriched yet
    to_process = [
        row for row in raw_rows
        if row["date"] not in already_done
    ]

    print(f"Raw records: {len(raw_rows)}")
    print(f"Already enriched: {len(already_done)}")
    print(f"Records to process: {len(to_process)}")

    return to_process, metadata

# =========================
# Step 2: ML Transform
# =========================

def ml_transform(to_process, metadata):
    # Load the trained ML model
    clf = joblib.load("models/weather_classifier.pkl")

    # Get the feature columns from the metadata
    features = metadata["features"]

    # Build a DataFrame from the unprocessed records
    df = pd.DataFrame(to_process)

    # Select features in the order expected by the model
    X = df[features]

    # Make predictions
    predictions = clf.predict(X)

    # Get prediction probabilities
    probabilities = clf.predict_proba(X)[:, 1]

    # Build enrichment records
    enrichment_records = []

    for i in range(len(to_process)):
        record = {
            "date": to_process[i]["date"],
            "good_for_running": bool(predictions[i]),
            "confidence": round(float(probabilities[i]), 4)
        }

        enrichment_records.append(record)

    # Count good-for-running days
    good_count = sum(
        record["good_for_running"]
        for record in enrichment_records
    )

    # Get confidence range
    confidence_min = min(probabilities)
    confidence_max = max(probabilities)

    print(f"Good days: {good_count}")
    print(f"Confidence range: {confidence_min:.2f} - {confidence_max:.2f}")

    return enrichment_records

# =========================
# Step 3: LLM Transform
# =========================

SYSTEM_PROMPT = (
    "You are writing a one-sentence running recommendation for a daily weather summary app. "
    "You will receive weather conditions for a single day and a machine learning prediction "
    "about whether the day is good for running. "
    "Write exactly one sentence — direct, practical, and specific to the conditions. "
    "Do not use bullet points, headers, or phrases like 'Based on the data'."
)

def make_user_message(row, good_for_running, confidence):
    prediction_text = (
        "good for running"
        if good_for_running
        else "not ideal for running"
    )

    return (
        f"Date: {row['date']}\n"
        f"High: {row['temperature_2m_max']}°C, "
        f"Low: {row['temperature_2m_min']}°C\n"
        f"Precipitation: {row['precipitation_sum']} mm\n"
        f"Max wind speed: {row['wind_speed_10m_max']} km/h\n"
        f"Model prediction: {prediction_text} "
        f"(confidence: {confidence:.0%})"
    )

def llm_transform(to_process, enrichment_records):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for i, record in enumerate(enrichment_records):
        raw_row = to_process[i]

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": make_user_message(
                            raw_row,
                            record["good_for_running"],
                            record["confidence"]
                        )
                    }
                ],
                max_tokens=100
            )

            summary = response.choices[0].message.content.strip()

            if summary:
                record["llm_summary"] = summary
            else:
                record["llm_summary"] = "Recommendation unavailable."

        except Exception as e:
            print(f"API error on {record['date']}: {e}")
            record["llm_summary"] = "Recommendation unavailable."

        if (i + 1) % 50 == 0:
            print(
                f"Processed {i + 1} / {len(enrichment_records)}"
            )

    return enrichment_records

# =========================
# Step 4: Load
# =========================

def load_enriched(supabase, enrichment_records):
    response = (
        supabase.table("weather_enriched")
        .upsert(enrichment_records, on_conflict="date")
        .execute()
    )

    print(f"Upserted {len(response.data)} rows into weather_enriched")

    return response.data


# =========================
# Step 5: Verify
# =========================

# Step 5: Verify

def verify_enriched(supabase):
    total = (
        supabase.table("weather_enriched")
        .select("date", count="exact")
        .execute()
    )

    print(f"Total rows: {total.count}")

    sample = (
        supabase.table("weather_enriched")
        .select("date, good_for_running, confidence, llm_summary")
        .limit(5)
        .execute()
    )

    print("\nSample rows:")

    for row in sample.data:
        print(
            f"{row['date']} | "
            f"good_for_running={row['good_for_running']} | "
            f"confidence={row['confidence']:.2f} | "
            f"llm_summary={row['llm_summary']}"
        )

    good_count = (
        supabase.table("weather_enriched")
        .select("date", count="exact")
        .eq("good_for_running", True)
        .execute()
    )

    print(f"\nGood-for-running days: {good_count.count}")

# LLM summary review:
# The summaries generally reflected the weather features and the model prediction.
# A particularly good summary was the January 2 summary because it mentioned
# the mild temperatures, no precipitation, light winds, and matched the model prediction.
# A weaker summary was the January 1 summary because it was less specific about
# the actual weather conditions and only generally mentioned mild temperatures,
# minimal precipitation, and wind.
# The recommendation was still reasonable, but it provided less specific information
# than some of the other summaries.

if __name__ == "__main__":
    supabase = get_client()
    to_process, metadata = incremental_read(supabase)

    if to_process:
        enrichment_records = ml_transform(to_process, metadata)
        enrichment_records = llm_transform(to_process, enrichment_records)
        load_enriched(supabase, enrichment_records)
    verify_enriched(supabase)

# =========================
# Step 6: Reflect
# =========================

# 1. The ML classifier was trained on Charlotte, NC data. If you loaded weather data for a different city in Week 9, do you expect the classifier's predictions to be accurate? Why or why not?
# If the classifier was trained on Charlotte, NC and used on data from a different city, its accuracy could be different because cities can have different weather patterns and conditions.
# The model may not perform as well on weather data that is different from the data it was trained on.

# 2. The LLM recommendations are generated from the model's prediction and the weather features. Does the LLM have any ability to "override" the classifier, or is it purely additive? What are the implications of that?
# The LLM does not override the classifier. It is an additive step that uses the classifier's prediction and the weather features to create a natural-language recommendation.
# This means the LLM explains the prediction but does not change the original ML prediction.

# 3. If you ran this pipeline on 50,000 records instead of 365, what would be your main concern: cost, latency, or something else? How would you address it?
# My main concerns would be API cost and latency because each record requires an LLM call.
# I could reduce these concerns by avoiding unnecessary calls, using batching or controlled concurrency, and using a lower-cost model when appropriate.
# I would also monitor the pipeline to make sure it is processing records efficiently and handling API errors correctly.