# --- Supabase Connection ---
# Q1

"""

1. The Supabase Project URL
2. The Supabase API key (anon key)

Both can be found in the Supabase dashboard under Project Settings > API.

They should not be hardcoded in a Python script because the credentials
could be exposed if the code is shared or uploaded to GitHub. Instead,
they should be stored in a .env file.

"""

# Q2

import os
from dotenv import load_dotenv
from supabase import create_client

def get_client():
    """
    Returns a Supabase client instance using the URL and API key from environment variables.
    """
    load_dotenv()  # Load environment variables from .env file

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    if not SUPABASE_URL:
        raise ValueError("Supabase URL is missing. Please check your .env file.")
    if not SUPABASE_KEY:
        raise ValueError("Supabase Key is missing. Please check your .env file.")

    return create_client(SUPABASE_URL, SUPABASE_KEY)

# Q3

"""
Connection Question 3:

Row Level Security (RLS) is a security feature in Supabase that
controls which rows users can access in a database table.

We disabled RLS on our tables for this course because this is a
learning project and we need simple access to our tables while
building and testing the pipeline.

In a real-world application, I would keep RLS enabled when the
database contains user-specific or sensitive data. For example,
in a banking or healthcare application, RLS could make sure that
users can only access their own records.

"""

# --- supabase-py CRUD ---
# Q1

from datetime import date

def insert_test_record(supabase):
    record = {
        "date": str(date.today()),
        "temperature_2m_max": 75.0,
        "temperature_2m_min": 55.0,
        "precipitation_sum": 0.5,
        "wind_speed_10m_max": 12.0
    }

    return supabase.table("weather_raw").insert(record).execute()

supabase = get_client()

response = insert_test_record(supabase)

print(response.data)

"""
The second run fails because the date column is the primary key,
and the same date cannot be inserted twice.
To make the function safe to run multiple times, I would use upsert()
with on_conflict="date". This will insert the record if the date does not
exist and update the record if the date already exists.
"""

# Q2

def get_records_by_date_range(supabase, start, end):
    response = supabase.table("weather_raw").select("*").gte("date", start).lte("date", end).execute()
    return response.data

start = "2026-09-01"
end = "2026-09-03"

records = get_records_by_date_range(supabase, start, end)

print("Retrieved:", records)


# Q3

"""
The difference between insert and upsert:

insert() adds a new row to the table. If a row with the same primary key
already exists, the insert will fail.

For example, I would use insert() when I know the record is new and I
do not want an existing record to be changed.

upsert() can insert a new row or update an existing row when there is a
conflict with the specified key.

For example, I would use upsert() in a weather data pipeline that may
run multiple times. If a date already exists, the existing weather
record can be updated instead of causing an error.
"""

def safe_upsert(supabase, records):
    response = (
        supabase.table("weather_raw")
        .upsert(records, on_conflict="date")
        .execute()
    )

    print("Rows affected:", len(response.data))

    return response.data


# --- Idempotency ---
# Q1

"""
Idempotency is important in a data pipeline because the pipeline may need
to be restarted after a failure. An idempotent pipeline can run multiple
times without creating duplicate or incorrect data.

For example, if a pipeline loads 100 weather records and crashes after
loading 50, restarting a non-idempotent pipeline could insert those same
50 records again, causing duplicates. Using upsert() with date as the
conflict key allows the pipeline to restart safely without creating
duplicate records.
"""

