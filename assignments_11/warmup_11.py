# --- Prefect Orchestration ---

# ============================================================
# Prefect Question 1
# ============================================================

# @task is used for individual units of work in a Prefect workflow.
# @flow is used to define and run the overall workflow and can call tasks.
#
# I would not use @task for a simple Celsius-to-Fahrenheit conversion.
# It is a pure in-memory calculation with no I/O, so a regular Python
# function is simpler and is enough.

# ============================================================
# Prefect Question 2
# ============================================================

@task(retries=3, retry_delay_seconds=30)

# ============================================================
# Prefect Question 3
# ============================================================

# I would look at the flow run in the Prefect UI and open the failed
# transform task.
#
# I would check the task logs and error message/traceback to see what
# caused the failure. I would also check the task state and details,
# including when it failed and any error information.
#
# Since transform failed, load_enriched did not run because it depends
# on the transform task.


# --- Production Patterns ---

# ============================================================
# Production Question 1
# ============================================================

# raise_for_status() checks the HTTP response status code and raises
# an exception when the response indicates an HTTP error, such as 500.
#
# This is better for a pipeline because the failure is explicit and
# Prefect can detect the failed task.
#
# With raise_for_status(), a 500 error raises an exception and the task
# fails, so downstream tasks will not run.
#
# With only print("error"), the task can continue and may be marked
# as successful, allowing downstream tasks to run with bad or missing data.

# ============================================================
# Production Question 2
# ============================================================

# upsert protects the pipeline from duplicate records when the pipeline
# is re-run. With on_conflict="date", an existing date is updated instead
# of creating a duplicate row.
#
# If plain insert were used, re-running from the beginning could cause
# duplicate-key errors for records that were already loaded. The pipeline
# could fail before completing.

# ============================================================
# Production Question 3
# ============================================================

@task
def load_enriched(enrichment_records: list):
    get_run_logger().info(f"Upserted {len(enrichment_records)} enrichment records.")

# ============================================================
# Production Question 4
# ============================================================

# The incremental processing check helps make the pipeline idempotent
# by processing only records that still need enrichment.
#
# Without the check, the ML and LLM steps would run on all 365 records
# every time. This would increase API/LLM costs and make the pipeline
# take longer.
#
# It could also create unnecessary updates or inconsistent results if
# the ML or LLM output changes between runs. The incremental check avoids
# unnecessary processing and helps keep existing enriched data stable.