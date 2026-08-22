## Part A: Supabase Setup

The Supabase project is set up successfully with the weather_raw and weather_enriched tables and the required columns.


## Part B: Cloud Cost Analysis
### Scenario A — Lightweight compute: A t3.micro EC2 instance (1 vCPU, 1 GB RAM), on-demand pricing, running 8 hours per day, 5 days per week (approximately 160 hours per month). Use the US East (N. Virginia) region.

Scenario A costs about $1.58 per month. The cost did not surprise me because it is a small instance with limited compute resources.

### Scenario B — Heavy analytics workload: A p3.2xlarge EC2 instance (8 vCPU, 1 V100 GPU), running 24/7 for the full month (730 hours); an RDS db.m5.large instance (2 vCPU, 8 GB RAM); and an S3 Standard storage bucket with 1 TB of data. Use US East (N. Virginia).

Scenario B costs about $2,438.94 per month. The cost surprised me because I can see how the cost can increase quickly by using more resources. 


### Anything interesting you found while exploring the calculator beyond the two required scenarios.

The cost increases based on the instance type, whether use a single or multiple Availability Zones, and the number of instances.

### A sentence on how the two scenarios compare — what does the cost difference tell you about when a GPU instance is or isn't worth it?

Compared with Scenario A, Scenario B is much more expensive because it uses a large GPU and runs 24 hours a day, every day. A GPU instance is worth the cost when a project requires high computing power, such as large data analytics or machine learning model training, but it may not be worth it for a small project that does not need large computing resources or a GPU.


## Here is the video link:
https://youtu.be/ZhfnyACX1_E
