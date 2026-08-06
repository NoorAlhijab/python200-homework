# --- Completions API ---
# ==========================
# API Question 1
# ==========================

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What is one thing that makes Python a good language for beginners?"}]
)

print("Response:")
print(response.choices[0].message.content)

print("\nModel:")
print(response.model)

# Print total number of tokens used
print("\nTotal number of tokens:")
print(response.usage.total_tokens)

# ==========================
# API Question 2: Temperature
# ==========================

prompt = "Suggest a creative name for a data engineering consultancy."
temperatures = [0, 0.7, 1.5]
for temp in temperatures:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=temp
    )
    print(f"\nTemperature: {temp}")
    print("Response:")
    print(response.choices[0].message.content)
    print("-" * 40)

 
 
# Responses became more creative as the temperature increased.
# Temperature 0 gave the most consistent answer, while temperature 1.5 gave the most varied answer.
# I would use temperature = 0 if I wanted consistent and reproducible output.

# ==========================
# API Question 3
# ==========================

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Give me a one-sentence fun fact about pandas (the animal, not the library)."}],
    n=3,
    temperature=1.0
) 
for i, choice in enumerate(response.choices, start=1):
    print(f"\nCompletion {i}:")
    print(choice.message.content)
    print("-" * 40)

# ==========================
# API Question 4
# ==========================

response = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=[{"role": "user", "content": "Explain how neural networks work."}],
    max_tokens=15
)

print("Response", response.choices[0].message.content)

# The response was incomplete because max_tokens was set to 15,
# which limited the number of tokens the model could generate.
# max_tokens is useful for controlling response length, reducing costs,
# and preventing unnecessarily long outputs.

# --- System Messages and Personas ---
# ==========================
# System Question 1
# ==========================

messages = [
    {"role": "system", "content": "You are a patient, encouraging Python tutor. You always explain things simply and end with a word of encouragement."},
    {"role": "user", "content": "I don't understand what a list comprehension is."}
]

response_1 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)

print("Python Tutor Response:\n", response_1.choices[0].message.content)

# Second personality
messages = [
    {"role": "system", "content": "You are a sarcastic, witty Python tutor. You explain things in a humorous way and often use sarcasm."},
    {"role": "user", "content": "I don't understand what a list comprehension is"}
]

response_2 = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)
print("Sarcastic Python Tutor Response:\n", response_2.choices[0].message.content)

# Changing the system message changed the personality of the response.
# The first response was friendly, simple, and encouraging.
# The second response was more sarcastic and humorous.


# ==========================
# System Question 2
# ==========================

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "My name is Jordan and I'm learning Python."},
    {"role": "assistant", "content": "Nice to meet you, Jordan! Python is a great choice. What would you like to work on?"},
    {"role": "user", "content": "Can you remind me what my name is?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages
)   
print("Response:\n", response.choices[0].message.content)
# The model knows Jordan's name because the conversation history was included
# in the request. The API is stateless, so it only knows what we send in the
# messages list.

# --- Prompt Engineering ---

# ==========================
# Prompt Question 1 — Zero-Shot
# ==========================

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Analyze the sentiment of the following customer reviews and classify each as positive, negative, or mixed.
Print each result labeled with the review number and the sentiment classification.

Reviews:
1. {reviews[0]} 
2. {reviews[1]}
3. {reviews[2]}
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Zero-Shot Sentiment Analysis:\n", response.choices[0].message.content)

# ==========================
# Prompt Question 2 — One-Shot
# ==========================

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Analyze the sentiment of the following customer reviews and classify each as positive, negative, or mixed.
Print each result labeled with the review number and the sentiment classification.

Example:
Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Reviews:
1. {reviews[0]} 
2. {reviews[1]}
3. {reviews[2]}
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("One-Shot Sentiment Analysis:\n", response.choices[0].message.content)

# Adding one example in the prompt improved the output format and consistency.
# The sentiment classifications stayed the same, but the model followed the Review/Sentiment format.

# ==========================
# Prompt Question 3 — Few-Shot
# ==========================

reviews = [
    "The onboarding process was smooth and the team was welcoming.",
    "The software crashes constantly and support never responds.",
    "Great price, but the documentation is nearly impossible to follow."
]

prompt = f"""
Analyze the sentiment of the following customer reviews and classify each as positive, negative, or mixed.
Print each result labeled with the review number and the sentiment classification.

Examples:

Review: "The product works perfectly and the customer service was excellent."
Sentiment: positive

Review: "The app is slow, full of bugs, and very frustrating to use."
Sentiment: negative

Review: "Fast shipping but the item arrived damaged."
Sentiment: mixed

Reviews:
1. {reviews[0]} 
2. {reviews[1]}
3. {reviews[2]}
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

print("Few-Shot Sentiment Analysis:\n", response.choices[0].message.content)

# Zero-shot is useful for simple tasks when the model already understands the instructions.
# One-shot is useful when you want to guide the output format with one example.
# Few-shot is useful when you need more consistency or when the task has specific patterns that examples can clarify.

# ==========================
# Prompt Question 4 — Chain of Thought
# ==========================

prompt = """
Solve the following math problem.

Show your reasoning step by step before giving the final answer.
Label the final answer clearly.

Problem:
A data engineer earns $85,000 per year. She gets a 12% raise, then 6 months later
takes a new job that pays $7,500 more per year than her post-raise salary.
What is her final annual salary?
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]  
)

print("Chain of Thought Response:\n", response.choices[0].message.content)

# The response was easier to follow because the problem was broken
# into smaller calculation steps before providing the final answer.

# ==========================
# Prompt Question 5 — Structured Output
# ==========================

import json

review = "I've been using this tool for three months. It handles large datasets well, \
but the UI is clunky and the export options are limited."

prompt = f"""
Analyze the following customer review.

Return ONLY valid JSON.
Do not include any explanation, markdown, or code fences.

The JSON must contain these keys:
- sentiment
- confidence (a float from 0 to 1)
- reason (one sentence)

Review:
"{review}"
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
raw_response = response.choices[0].message.content
print("Raw Response:\n", raw_response)

try:
    result = json.loads(raw_response)

    print("Sentiment:", result["sentiment"])
    print("Confidence:", result["confidence"])
    print("Reason:", result["reason"])

except json.JSONDecodeError:
    print("Raw response was not valid JSON:")
    print(raw_response)

# ==========================
# Prompt Question 6 — Delimiters
# ==========================

user_text = "First boil a pot of water. Once boiling, add a handful of salt and the \
pasta. Cook for 8-10 minutes until al dente. Drain and toss with your sauce of choice."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)   

print("Instructions Response:\n", response.choices[0].message.content)

# Second prompt (not instructions)
user_text = "The weather today is sunny with a high of 75 degrees. There is a light breeze and no chance of rain."

prompt = f"""
You will be given text inside triple backticks.
If it contains step-by-step instructions, rewrite them as a numbered list.
If it does not contain instructions, respond with exactly: "No steps provided."

```{user_text}```
"""
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("No Instructions Response:\n", response.choices[0].message.content)
# Delimiters clearly separate the user's text from the instructions.
# This helps prevent the model from confusing the prompt with the input text.


# --- Local Models with Ollama ---

# ==========================
# Ollama Question 1
# ==========================

# Ollama terminal output:
"""
>>> ollama run qwen3:0.6b

A large language model is an AI system trained on vast amounts of text to
understand and generate human-like language. It learns patterns from data
to produce responses based on user prompts.
"""

prompt = "Explain what a large language model is in two sentences."
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)
print("OpenAI Response:\n", response.choices[0].message.content)

"""
OpenAI output:

 A large language model is an artificial intelligence system designed to understand and generate human-like text 
 by analyzing vast amounts of text data. It uses deep learning techniques to predict and produce coherent responses 
 based on the context of the input it receives.

"""
# Comparison:
# Both models explained what an LLM is. The Ollama response was shorter,
# while OpenAI provided a more detailed explanation. Ollama's advantage is
# privacy because it runs locally. A disadvantage is that local models may
# require more hardware and may not perform as well as larger cloud models.

