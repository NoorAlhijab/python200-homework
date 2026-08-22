## Part 1: Warmup — Cloud Concepts
## Cloud Concepts Question 1
### What is the core economic model of cloud computing, and how does it differ from owning your own servers?

The core concept in cloud computing is that you rent resources and pay for what you use, known as pay-as-you-go. So instead of paying for and maintaining your own resources, you use resources provided by the cloud provider.


## Cloud Concepts Question 2
### What is the difference between vertical scaling and horizontal scaling? Give a concrete example of when you might choose each.

Vertical scaling means increasing the size of CPU, GPU, and RAM, while horizontal scaling means increasing the number of virtual machines.

### Then, for the three scenarios below, write one sentence saying which type of scaling applies and why.

### A web app that normally handles 1,000 users per day suddenly needs to handle 100,000 after a viral product launch.

Horizontal scaling fits this scenario since it needs to scale on demand.

### A data scientist's model training job is running too slowly, and they want a machine with a faster GPU and more RAM.

Vertical scaling fits this scenario since it needs to increase the GPU and RAM.

### A data pipeline that processes 10 files per run now needs to process 10,000 files per run, and the work can be split across machines.

Horizontal scaling fits this scenario since it needs to increase the number of virtual machines.


## Cloud Concepts Question 3

### Before writing your definitions, classify each item in the list below as IaaS, PaaS, SaaS, or BaaS. One sentence of reasoning is enough for each.

Gmail: SaaS because it's a complete application that customers can use through the internet.

Azure Virtual Machines: IaaS because it's a virtual machine that comes with computing and networking resources. Customers are responsible for setting up the operating system and software and applying security updates.

AWS S3 (Simple Storage Service): IaaS because it's cloud object storage used to store files in buckets.

GitHub Codespaces: PaaS because it provides a managed development environment for writing and running code.

Snowflake: PaaS because it provides a managed cloud data platform.

Supabase: BaaS because it provides backend services such as a database and authentication.

### Now describe IaaS, PaaS, and SaaS in your own words. For each, give one example (from the lesson or the list above) and describe what you, as the developer, are responsible for managing.

IaaS is Infrastructure as a Service. Customers rent infrastructure from a cloud provider, such as virtual machines. Customers are responsible for managing the operating system, software, and configurations. Example: EC2 in AWS and Azure Virtual Machines in Microsoft Azure.

PaaS is Platform as a Service. Customers use a managed platform to develop and run applications without managing the underlying infrastructure. Example: GitHub Codespaces and AWS Elastic Beanstalk.

SaaS is Software as a Service. Customers use a complete software application that is ready to use through the internet without managing the underlying infrastructure or software. Example: Gmail and Zoom.

## Cloud Concepts Question 4
### What is a managed data platform like Databricks or Snowflake, and how does it differ from using a cloud provider like AWS or GCP directly? What do you gain, and what do you give up?

A managed data platform is an application that is ready to work with data without requiring you to handle the setup and management of the infrastructure. When using a cloud provider like AWS or GCP directly, you need to set up the infrastructure and connect the resources needed to work with data.

You gain fast environment setup, but you give up some control over the infrastructure.

## Cloud Concepts Question 5
## The lesson names two situations where the cloud is probably not the right choice. What are they?

If the dataset fits comfortably on a single machine and does not have massive compute demands, local processing is often faster and cheaper. This is often the best approach when setting up an initial prototype.

The learning curve for cloud infrastructure can be very steep. Even doing simple things in the cloud can take a long time, as you have to figure out the right resources and jargon initially.
 

## Part 2: Warmup — Cloud Landscape
## Cloud Landscape Question 1
### Name the three hyperscalers. For each, write one sentence describing its primary strength and the type of organization most likely to use it.

Amazon Web Services (AWS): Its primary strength is infrastructure.
Google Cloud Platform (GCP): Its primary strength is data and machine learning.
Microsoft Azure: Its primary strength is enterprise and government settings.

## Cloud Landscape Question 2
### The lesson explains why this course switched from Microsoft Azure to Supabase. It gives three concrete reasons. Summarize each reason in your own words — one sentence each.

1. Access: Azure requires setup steps that may consume more time and add complexity.

2. Pedagogical fit: Azure uses a service named Azure Blob Storage to store files by path, which does not work with tables and rows like a relational database.

3. Pipeline coherence: Supabase streamlines the use of a database that works with tables and rows, which fits the ETL pipeline built in this course.

### Then add your own reflection: what does this suggest about how you should evaluate a cloud tool when starting a new project?

When working on a real-world project, you don't need to stick to one provider. Instead, use the right fit for each phase of the project.

## Cloud Landscape Question 3
### For each of the four scenarios below, identify which service category from the taxonomy table applies (e.g., "object storage", "managed relational DB", "LLM API", "serverless compute") and name one specific provider or product that offers it.

1. You need to store 10 TB of image files and retrieve them by filename from any machine.

Object storage: AWS S3 from AWS provider.

2. You need to run an ML training job on a GPU for four hours, then shut it down.

ML platform: Vertex AI from Google Cloud Platform (GCP) provider.

3. You need to host a web API that automatically scales up when traffic spikes and scales down when it quiets.

Serverless compute: AWS Lambda from AWS provider.

4. You need to send structured data to a large language model and get a text response back.

LLM API: OpenAI API.

## Cloud Landscape Question 4
### The lesson says most projects don't use one provider for everything. Describe a simple data project of your own design (one or two sentences is fine) and sketch a plausible stack using services from at least two different providers or products from the taxonomy table. Then answer: is there a benefit to consolidating to one provider, and what would you give up if you did?

A data project could use Supabase for a managed relational database and GCP BigQuery for data analytics.

Using one provider can simplify management, integration, and security, but you may be limited in choosing the best service from different providers based on factors like price.
