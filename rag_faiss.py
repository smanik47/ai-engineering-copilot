import os
import json
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------
# INITIALIZATION
# --------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

KNOWLEDGE_DIR = Path("knowledge")

INDEX_PATH = KNOWLEDGE_DIR / "faiss.index"
CHUNKS_PATH = KNOWLEDGE_DIR / "chunks.json"


# --------------------------------
# STEP 1: LOAD DOCUMENT
# --------------------------------

def load_document(file_path: str):

    return Path(file_path).read_text(encoding="utf-8")


# --------------------------------
# STEP 2: CHUNK DOCUMENT
# --------------------------------

def split_into_chunks(text, chunk_size=300, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start = end - overlap

    return chunks


# --------------------------------
# STEP 3: GENERATE EMBEDDINGS
# --------------------------------

def generate_embeddings(texts):

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
# STEP 4: BUILD FAISS INDEX
# --------------------------------

def build_faiss_index(chunks):

    embeddings = generate_embeddings(chunks)

    # Convert Python list into NumPy array
    vectors = np.array(
        embeddings,
        dtype="float32"
    )

    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)

    # Get embedding dimensions
    dimension = vectors.shape[1]

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)

    # Add document vectors
    index.add(vectors)

    # Persist index and chunk mapping
    faiss.write_index(index, str(INDEX_PATH))

    CHUNKS_PATH.write_text(
        json.dumps(chunks, ensure_ascii=False),
        encoding="utf-8"
    )

    return index, chunks


# --------------------------------
# STEP 5: LOAD EXISTING INDEX
# --------------------------------

def load_faiss_index():

    index = faiss.read_index(str(INDEX_PATH))

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    return index, chunks


# --------------------------------
# STEP 6: RETRIEVE RELEVANT CHUNKS
# --------------------------------

def retrieve_relevant_chunks(question, index, chunks, top_k=3):

    query_embedding = generate_embeddings([question])

    query_vector = np.array(
        query_embedding,
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            "chunk": chunks[idx],
            "score": float(score)
        })

    return results


# --------------------------------
# STEP 7: GENERATE ANSWER
# --------------------------------

def generate_answer(question, retrieved_chunks):

    context = "\n\n".join(
        item["chunk"]
        for item in retrieved_chunks
    )

    prompt = f"""
You are a technical assistant.

Answer the question using only the context below.

If the context does not contain enough information,
say that you do not have enough information.

Context:
{context}

Question:
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

if INDEX_PATH.exists() and CHUNKS_PATH.exists():

    print("Loading existing FAISS index...")

    index, chunks = load_faiss_index()

else:

    print("Creating FAISS index...")

    document = load_document(
        "knowledge/database.txt"
    )

    chunks = split_into_chunks(document)

    index, chunks = build_faiss_index(chunks)


print(f"Total vectors stored: {index.ntotal}")

question = input("\nAsk your question: ")

retrieved_chunks = retrieve_relevant_chunks(
    question,
    index,
    chunks,
    top_k=3
)

print("\nRetrieved chunks:")

for item in retrieved_chunks:

    print(f"\nSimilarity: {item['score']:.4f}")
    print(item["chunk"])


answer = generate_answer(
    question,
    retrieved_chunks
)

print("\nAI Generated Answer:")
print(answer)