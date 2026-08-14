# --- RAG Concepts ---
from dotenv import load_dotenv
import os

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")


# --- RAG Concepts ---

# =====================
# Concepts Question 1
# =====================

# Scenario A:
# Using a RAG (Retrieval-Augmented Generation) approach, the assistant can retrieve
# relevant information from the internal policy library and generate accurate responses 
# and the assistant can use the most recent information
# without needing to retrain the model every time the documents are updated.


# Scenario B:
# Using fine-tuning because the startup has 3,000 examples of their own product copy.
# The model can learn the specific brand voice from these examples and generate product
# copy in the same dry and minimalist style.


# Scenario C:
# Using prompt engineering since it is simple and only uses one two-page report.
# The report can be included in the prompt so the LLM can answer questions about it.

# =====================
# Concepts Question 2
# =====================

# AI models sometimes generate hallucination responses that can mislead users
# by providing wrong information. A confidently wrong answer is more harmful because
# users may trust the answer and act on it when the model sounds very sure.
# For example, if an AI gives wrong medical information about how to take a medication,
# a person could follow the advice and get harmed.

# =====================
# Concepts Question 3
# =====================

# The original list provided in the assignment:
#
# steps = [
#     "Generate a response from the LLM",
#     "Extract text from source documents",
#     "Receive the user's query",
#     "Retrieve the most relevant chunks",
#     "Convert text chunks into embeddings",
#     "Inject retrieved chunks into the prompt",
#     "Split text into chunks",
#     "Embed the user's query",
# ]

# Correct RAG steps 
#
# steps = [
#     "Extract text from source documents",
#     "Split text into chunks",
#     "Convert text chunks into embeddings",
#     "Receive the user's query",
#     "Embed the user's query",
#     "Retrieve the most relevant chunks",
#     "Inject retrieved chunks into the prompt",
#     "Generate a response from the LLM",
# ]
#
# 1. Extract text from source documents
# Get the text from the source documents so it can be processed.
#
# 2. Split text into chunks
# Break the text into smaller chunks so the relevant information can be found.
#
# 3. Convert text chunks into embeddings
# Convert each text chunk into an embedding so the meaning of the text can be compared.
#
# 4. Receive the user's query
# Receive the question from the user.
#
# 5. Embed the user's query
# Convert the user's question into an embedding so it can be compared with the document chunks.
#
# 6. Retrieve the most relevant chunks
# Find the chunks that are most relevant to the user's question.
#
# 7. Inject retrieved chunks into the prompt
# Add the relevant chunks to the prompt so the LLM has the information it needs.
#
# 8. Generate a response from the LLM
# The LLM uses the user's question and the retrieved information to generate a response.



# --- Keyword RAG ---

import string

def simple_keyword_retrieval(query, documents, verbose=True):
    """Keyword retrieval using token overlap scoring."""
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "of", "for", "to", "is",
        "are", "was", "were", "by", "with", "at", "from", "that", "this",
        "as", "be", "it", "its", "their", "they", "we", "you", "our", "your"
    }
    translator = str.maketrans("", "", string.punctuation)

    query_words = {
        w.translate(translator)
        for w in query.lower().split()
        if w not in stopwords
    }
    if verbose:
        print(f"\nQuery tokens (filtered): {sorted(query_words)}")

    scores = []
    for name, content in documents.items():
        content_words = {
            w.translate(translator)
            for w in content.lower().split()
            if w not in stopwords
        }
        overlap = query_words & content_words
        score = len(overlap)
        scores.append((score, name, content))
        if verbose:
            print(f"[{name}] overlap={score} -> {sorted(overlap)}")

    scores.sort(reverse=True)
    best = next(((name, content) for score, name, content in scores if score > 0), None)
    if best:
        if verbose:
            print(f"\nSelected best match: {best[0]}")
        return [best]
    else:
        if verbose:
            print("\nNo overlapping keywords found.")
        return [("None found", "No relevant content.")]

# =====================
# Keyword Question 1
# =====================

query = "What are your hours on weekends?"

documents = {
    "menu.txt": "We serve espresso, lattes, cappuccinos, and cold brew. Pastries include croissants and muffins baked fresh daily. Oat milk and almond milk are available.",
    "hours.txt": "We are open Monday through Friday from 7am to 7pm. On weekends we open at 8am and close at 5pm. We are closed on Thanksgiving and Christmas Day.",
    "hiring.txt": "We are currently hiring baristas and shift supervisors. Send your resume to jobs@groundworkcoffee.com.",
    "loyalty.txt": "Join our loyalty program to earn one point per dollar spent. Redeem 100 points for a free drink of your choice.",
}

results = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", results[0][0])

# The result was hours.txt, which is the correct document for this question.
# The stop-word filtering removes common words that do not provide useful
# information for retrieval. The remaining relevant keywords include
# "hours" and "weekends", and "weekends" matches the hours.txt document.
# Therefore, the keyword retriever correctly selects hours.txt.

# =====================
# Keyword Question 2
# =====================

query = "Do you have anything without caffeine?"

results = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", results[0][0])

# The result was "Selected document: None found" because there were no
# overlapping keywords between the query and any of the documents.
# Keyword retrieval only looks for exact keyword matches, so it does not
# understand that "without caffeine" could relate to drinks such as coffee.
# Semantic retrieval could perform better because it compares the meaning
# of the query with the meaning of the documents rather than only matching
# exact words.

# =====================
# Keyword Question 3
# =====================

# I think no document will be selected because the query does not have matching keywords in the documents.

query = "How do I sign up for rewards?"
results = simple_keyword_retrieval(query, documents, verbose=True)
print("Selected document:", results[0][0])

# My prediction was correct. The result was "None found" because the keyword retrieval looks for exact matching words.

# --- Semantic RAG Concepts ---

# =====================
# Semantic Question 1
# =====================

# 1. What is a vector embedding?
# A vector embedding converts text into a list of numbers that represents
# the meaning of the text. This allows a computer to compare the meaning
# of different pieces of text.

# 2. Which chunk is more relevant?
# The chunk with a cosine similarity score of 0.85 is more relevant.
# A higher score means the meaning of the chunk is more similar to the query.

# 3. Why can semantic search find relevant chunks without exact words?
# Semantic search looks at the meaning of the text instead of only matching
# exact words. This allows it to find text with similar meaning even when
# different words are used.

# =====================
# Semantic Question 2
# =====================

# | Feature                    | Keyword RAG                       | Semantic RAG                    |
# |----------------------------|-----------------------------------|---------------------------------|
# | What is compared?          | Exact word overlap                | Meaning  of the text            |
# | What is retrieved?         | Full document                     | Relevant text chunks            |
# | Can it handle synonyms?    | No                                | Yes                             |
# | Storage format             | Plain text dictionary             | Vector embeddings               |
# | Relevance score            | Number of overlapping keywords    | Cosine similarity score         |


# --- LlamaIndex ---

# =====================
# LlamaIndex Question 1
# =====================

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

docs = SimpleDirectoryReader(
    "../resources/brightleaf_pdfs"
).load_data()

print(f"Loaded {len(docs)} documents/pages")

# Build a vector index automatically (handles chunking + embeddings)
index = VectorStoreIndex.from_documents(docs)

query_engine = index.as_query_engine(similarity_top_k=3)

questions = [
    "What employee benefits does BrightLeaf offer?",
    "What are BrightLeaf's security policies?",
]


for q in questions:
    print(f"\nQ: {q}")
    response = query_engine.query(q)
    print("A:", response)

    for i, node_with_score in enumerate(response.source_nodes[:3], start=1):
        print(f"\nSource Node {i}:")
        print(f"Similarity Score: {node_with_score.score:.4f}")
        print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")

# Observation for Query 1:
# The first retrieved chunk was the most relevant because it contained
# information about BrightLeaf's employee benefits. The other two chunks
# were less relevant because they discussed the company mission and security
# policies. The response was specific and directly answered the question.
# This shows that the retriever can find the relevant information, but
# retrieving three chunks can also include some less relevant context.


# Observation for Query 2:
# The first retrieved chunk was the most relevant because it contained
# information about BrightLeaf's security policies. The other two chunks
# were less relevant because they discussed employee benefits and the company
# mission. The response was specific and directly answered the question.
# This shows that the retriever identified the appropriate security-related
# information, while also returning some unrelated context.

# =====================
# LlamaIndex Question 2
# =====================

question = "What employee benefits does BrightLeaf offer?"

# Run with similarity_top_k=1
query_engine_1 = index.as_query_engine(similarity_top_k=1)
response_1 = query_engine_1.query(question)

print("\n--- similarity_top_k=1 ---")
print("Q:", question)
print("A:", response_1)

for node_with_score in response_1.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")


# Run with similarity_top_k=5
query_engine_5 = index.as_query_engine(similarity_top_k=5)
response_5 = query_engine_5.query(question)

print("\n--- similarity_top_k=5 ---")
print("Q:", question)
print("A:", response_5)

for node_with_score in response_5.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")

# The answers were very similar for similarity_top_k=1 and similarity_top_k=5.
# With top_k=1, the response used only the highest-scoring and most relevant
# source chunk. With top_k=5, additional chunks were retrieved, but the
# additional chunks had lower similarity scores and were less relevant to
# the question. In this case, the extra context did not significantly change
# the answer. This shows that more retrieved context is not always better
# because lower-quality or less relevant chunks can add noise to the context.

# =====================
# LlamaIndex Question 3
# =====================

question = "What is BrightLeaf's employee satisfaction score?"

query_engine = index.as_query_engine(similarity_top_k=3)
response = query_engine.query(question)

print("\n--- LlamaIndex Question 3 ---")
print("Q:", question)
print("A:", response)

for node_with_score in response.source_nodes:
    print(f"Similarity Score: {node_with_score.score:.4f}")
    print(f"Text Snippet: {node_with_score.node.get_content()[:150]}...")
    print("-" * 30)

# I expected the pipeline to struggle because the documents do not contain
# an employee satisfaction score. The model correctly said that the score
# was not mentioned in the provided context. The retrieved chunks were
# related to employee well-being, security, and the company mission, but
# none contained the requested score. To improve the system, I would add
# a relevance threshold so the system can avoid using unrelated chunks
# when the requested information is not available.

# =====================
# LlamaIndex Question 4
# =====================
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator

# Create Judge LLM
llm = OpenAI(model="gpt-4o-mini", temperature=0.2)

# Define evaluators
faithfulness_evaluator = FaithfulnessEvaluator(llm=llm)
relevancy_evaluator = RelevancyEvaluator(llm=llm)


# First query
q = "What employee benefits does BrightLeaf offer?"
response = query_engine.query(q)

faithfulness_result = faithfulness_evaluator.evaluate_response(
    query=q,
    response=response
)

relevancy_result = relevancy_evaluator.evaluate_response(
    query=q,
    response=response
)

print("\n--- Employee Benefits Query ---")
print("Q:", q)
print("Response:", response)
print("Faithfulness Evaluation:", faithfulness_result.score)
print("Relevancy Evaluation:", relevancy_result.score)


# Second query: information not in the documents
q2 = "What is BrightLeaf's employee satisfaction score?"
response2 = query_engine.query(q2)

faithfulness_result2 = faithfulness_evaluator.evaluate_response(
    query=q2,
    response=response2
)

relevancy_result2 = relevancy_evaluator.evaluate_response(
    query=q2,
    response=response2
)

print("\n--- Missing Information Query ---")
print("Q:", q2)
print("Response:", response2)
print("Faithfulness Evaluation:", faithfulness_result2.score)
print("Relevancy Evaluation:", relevancy_result2.score)

# Both queries received 1.0 for faithfulness and relevancy.
#
# A faithfulness score of 1.0 means the answer is fully supported by
# the retrieved context. A score of 0.0 means the answer is not supported.
#
# A relevancy score measures whether the answer addresses the question.
# Faithfulness checks whether the response is supported by the context,
# while relevancy checks whether the response addresses the question.
#
# The scores did not change because the model correctly stated that
# the employee satisfaction score was not found in the provided documents.
# The model did not invent a score, so the response remained faithful to
# the available context and relevant to the question.
#
# LLM-as-a-judge means using an LLM to evaluate another LLM's answer.
# It is useful for RAG because it can check support and relevance,
# not just whether an answer matches one expected answer.