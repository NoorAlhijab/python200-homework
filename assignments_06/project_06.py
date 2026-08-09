# ===========================
# Step 1: Setup
# ===========================
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

if load_dotenv():
    print("API key loaded successfully.")
else:
    print("Warning: could not load API key. Check your .env file.")

# Check that the Groundwork documents directory exists
docs_dir = Path("assignments_06/resources/groundwork_docs")
assert docs_dir.exists(), f"Document directory not found: {docs_dir}"

print(f"Groundwork documents found: {docs_dir}")


# =============================
# Step 2: Load the Documents
# ============================

documents = SimpleDirectoryReader(
    input_dir=str(docs_dir)
).load_data()

print(f"Number of documents loaded: {len(documents)}")

# Get document metadata
for document in documents:
    print(f"File name: {document.metadata['file_name']}")

# ==========================================
# Step 3: Build the Index and Query Engine
# ==========================================

index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine(
    similarity_top_k=3
)

print("Index built successfully. Ready to answer questions.")

# =============================
# Step 4: Query the Assistant
# =============================

questions = [
    "What are Groundwork's hours on weekends?",
    "Do you offer any dairy-free milk options?",
    "How does the loyalty program work?",
    "How did Groundwork Coffee get started?",
    "Do you offer catering or wholesale orders?",
]

for question in questions:
    print("\n" + "=" * 60)
    print(f"Question: {question}")

    # Send the question to the query engine
    response = query_engine.query(question)

    # Print the answer
    print(f"Answer: {response}")

    # Get the top retrieved source node
    if response.source_nodes:
        top_node = response.source_nodes[0]

        print("\nTop Retrieved Source:")
        print(f"Document: {top_node.node.metadata['file_name']}")
        print(f"Similarity score: {top_node.score}")
        print(f"Chunk text: {top_node.node.get_content()[:200]}")
    else:
        print("\nNo source nodes were retrieved.")

# The assistant's responses were confident and mostly accurate.
# The answers were supported by information retrieved from the Groundwork documents.
# I noticed that the top retrieved document was not always the document I expected.
# For example, the dairy-free milk question retrieved seasonal_specials.txt,
# while the loyalty question retrieved faq.txt. However, the model still provided
# relevant and accurate answers. The story and catering questions had especially
# strong retrieval scores and clearly matched their source documents.

# =========================
# Step 5: Find a Failure
# =========================
        

failure_question = "What is Groundwork Coffee's annual revenue?"

print("\n" + "=" * 60)
print(f"Failure Question: {failure_question}")

# Query the assistant
failure_response = query_engine.query(failure_question)

# Print the full response
print(f"\nFull Response: {failure_response}")

# Print all three retrieved source nodes
print("\nAll Three Retrieved Source Nodes:")

for i, source_node in enumerate(failure_response.source_nodes[:3], start=1):
    print(f"\nSource Node {i}:")
    print(f"Document: {source_node.node.metadata['file_name']}")
    print(f"Similarity score: {source_node.score}")
    print(f"Chunk text: {source_node.node.get_content()[:200]}")

# I asked about Groundwork's annual revenue because it is not in the documents.
# The retrieval returned related documents, but none contained revenue information.
# The model correctly said the revenue could not be determined instead of guessing.
# This shows that AI responses should be checked against the retrieved sources.
# I would improve the system by adding a similarity threshold and requiring the
# assistant to say when the available information is not enough to answer.

# ======================
# Step 6: Reflection
# ======================

# 1. The LlamaIndex implementation took about 6 core lines of code to build
# the RAG system. This shows that using a framework can simplify complex
# tasks and reduce the amount of code developers need to write.
#  
# 2. A useful use case would be an e-commerce customer support assistant.
# It could answer customer questions using product information, return
# policies, shipping information, and other company documents.
# 3. One failure mode that RAG cannot fully prevent is a faithfulness failure.
# Even when the correct information is retrieved, the model can still
# misunderstand or incorrectly use that information when generating its answer.
