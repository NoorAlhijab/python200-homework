# Here is the video link: https://youtu.be/Ku16bJEEBf8

import requests
import os
from dotenv import load_dotenv
from supabase import create_client
from datetime import date

def get_client():
    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")
    
    if not SUPABASE_URL:
        raise ValueError("Supabase URL is missing. Please check your .env file.")
    if not SUPABASE_KEY:
        raise ValueError("Supabase Key is missing. Please check your .env file.")
    
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = get_client()

# Step 1: Extract

def extract_weather():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": 38.9072,
        "longitude": -77.0369,
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "timezone": "America/New_York"
    }

    response = requests.get(url, params=params)

    response.raise_for_status()

    data = response.json()

    print("API response received successfully.")
    print("Response keys:", data.keys())

    return data

# Step 2: Transform

def transform_weather(data):
    daily = data["daily"]

    records = []
    for i in range(len(daily["time"])):
        record = {
            "date": daily["time"][i],
            "temperature_2m_max": daily["temperature_2m_max"][i],
            "temperature_2m_min": daily["temperature_2m_min"][i],
            "precipitation_sum": daily["precipitation_sum"][i],
            "wind_speed_10m_max": daily["wind_speed_10m_max"][i]
        }
        records.append(record)
    print("First record:", records[0])
    print("Last record:", records[-1])
    print("Number of records:", len(records))
    return records    

"""
Expect 365 records for 2023 because 2023 was not a leap year.
If the number differs, there may be missing dates in the API response
or an incorrect start or end date.
"""


# Step 3: Load

def load_weather(supabase, records):
    response = (supabase.table("weather_raw").upsert(records, on_conflict="date").execute())
    print("Rows upserted:", len(response.data))
    return response.data

# Step 4: Verify

def verify_load(supabase):
    response = supabase.table("weather_raw").select("*", count="exact").execute()
    print("Total rows:", response.count)

    earliest = (supabase.table("weather_raw").select("date").order("date", desc=False).limit(1).execute())
    latest = (supabase.table("weather_raw").select("date").order("date", desc=True).limit(1).execute())

    print("Earliest date:", earliest.data[0]["date"])
    print("Latest date:", latest.data[0]["date"])

    july_4 = (
        supabase.table("weather_raw")
        .select("*")
        .eq("date", "2023-07-04")
        .execute()
    )

    if july_4.data:
        print("2023-07-04:", july_4.data[0])
    else:
        nearby = (
            supabase.table("weather_raw")
            .select("*")
            .gte("date", "2023-07-01")
            .lte("date", "2023-07-07")
            .execute()
        )

        if nearby.data:
            target_date = date(2023, 7, 4)

            nearest = min(
                nearby.data,
                key=lambda row: abs(
                    date.fromisoformat(row["date"]) - target_date
                )
            )

            print("July 4 was missing. Nearest date:", nearest)
        else:
            print("No nearby dates found.")

"""
Idempotency confirmation:

I ran the script twice. The first run loaded the 2023 weather records.
I ran the script a second time, and the total number of rows remained
the same because the records are upserted using date as the conflict key.

This confirms that the pipeline is idempotent and does not create
duplicate rows when it is run multiple times.
"""            

# --- Run Pipeline ---

data = extract_weather()

records = transform_weather(data)

load_weather(supabase, records)

verify_load(supabase)    