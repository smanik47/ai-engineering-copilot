import os
import math
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------
# STEP 1: Load the document
# --------------------------------

def load_document(file_path: str):

    return Path(file_path).read_text(encoding="utf-8")


# --------------------------------
# STEP 2: Split document into chunks
# --------------------------------

def split_into_chunks(text: str, chunk_size=300, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap

    return chunks


# --------------------------------
# STEP 3: Generate embeddings
# --------------------------------

def generate_embeddings(texts: list[str]):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    results = sorted(
        response.data,
        key=lambda item: item.index
    )

    return [item.embedding for item in results]


# --------------------------------
# STEP 4: Calculate cosine similarity
# --------------------------------

def cosine_similarity(vector_a, vector_b):

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(
        sum(a * a for a in vector_a)
    )

    magnitude_b = math.sqrt(
        sum(b * b for b in vector_b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0

    return dot_product / (magnitude_a * magnitude_b)


# --------------------------------
# STEP 5: Retrieve relevant chunks
# --------------------------------

def retrieve_relevant_chunks(
        question,
        chunks,
        chunk_embeddings,
        top_k=3
):

    query_embedding = generate_embeddings([question])[0]

    results = []

    for chunk, embedding in zip(chunks, chunk_embeddings):

        score = cosine_similarity(
            query_embedding,
            embedding
        )

        results.append((chunk, score))

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    return results[:top_k]


# --------------------------------
# STEP 6: Generate answer using LLM
# --------------------------------

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        chunk for chunk, score in retrieved_chunks
    )

    prompt = f"""
You are a technical assistant.

Answer the user's question using only the context provided below.

If the context does not contain enough information,
clearly say that you do not have enough information.

Context:
{context}

User question:
{question}

Answer:
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text


# --------------------------------
# MAIN EXECUTION
# --------------------------------

document = load_document("knowledge/database.txt")

chunks = split_into_chunks(
    document,
    chunk_size=300,
    overlap=50
)

print(f"Total chunks created: {len(chunks)}")

chunk_embeddings = generate_embeddings(chunks)

question = input("\nAsk your question: ")

retrieved_chunks = retrieve_relevant_chunks(
    question,
    chunks,
    chunk_embeddings,
    top_k=3
)

print("\nRetrieved context:")

for chunk, score in retrieved_chunks:

    print(f"\nSimilarity score: {score:.4f}")
    print(chunk)

answer = generate_answer(
    question,
    retrieved_chunks
)

print("\nAI Generated Answer:")
print(answer)