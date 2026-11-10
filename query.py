import json
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI


# --------------------------------
# INITIALIZATION
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "knowledge" / "processed"

INDEX_PATH = PROCESSED_DIR / "faiss.index"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

load_dotenv(BASE_DIR / ".env")

client = OpenAI()


# --------------------------------
# STEP 1: LOAD INDEX AND CHUNKS
# --------------------------------

def load_knowledge_base():

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            "Knowledge base not found. Run ingestion.py first."
        )

    index = faiss.read_index(str(INDEX_PATH))

    chunks = json.loads(
        CHUNKS_PATH.read_text(encoding="utf-8")
    )

    if index.ntotal != len(chunks):
        raise ValueError(
            "FAISS index and chunk metadata are out of sync."
        )

    return index, chunks


# --------------------------------
# STEP 2: GENERATE QUERY EMBEDDING
# --------------------------------

def generate_query_embedding(question):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[question]
    )

    vector = np.array(
        [response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(vector)

    return vector


# --------------------------------
# STEP 3: RETRIEVE RELEVANT CHUNKS
# --------------------------------

def retrieve_chunks(question, index, chunks, top_k=3):

    query_vector = generate_query_embedding(question)

    scores, indices = index.search(
        query_vector,
        min(top_k, index.ntotal)
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        results.append({
            **chunks[idx],
            "score": float(score)
        })

    return results


# --------------------------------
# STEP 4: GENERATE ANSWER
# --------------------------------

def generate_answer(question, retrieved_chunks):

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
Source: {item['source']}
Page: {item['page']}

Content:
{item['text']}
"""
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
You are a technical knowledge assistant.

Answer the user's question using the provided context.

Rules:
1. Use the supplied context as the source of truth.
2. Do not invent facts that are not supported by the context.
3. If the context is insufficient, say so clearly.
4. Mention the source filename and page number
   when using information from a document.

Context:
{context}

User question:
{question}
"""

    response = client.responses.create(
        model="gpt-5.5",
        input=prompt
    )

    return response.output_text


# --------------------------------
# MAIN EXECUTION
# --------------------------------

def main():

    index, chunks = load_knowledge_base()

    print(f"Knowledge base loaded: {index.ntotal} chunks")

    question = input("\nAsk your question: ")

    retrieved_chunks = retrieve_chunks(
        question,
        index,
        chunks,
        top_k=top_k
    )

    print("\nRetrieved sources:")

    for item in retrieved_chunks:

        print(
            f"\nSource: {item['source']}"
            f"\nPage: {item['page']}"
            f"\nSimilarity: {item['score']:.4f}"
            f"\nText: {item['text']}"
        )

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    print("\nAI Generated Answer:")
    print(answer)


if __name__ == "__main__":
    main()

def answer_question(question, top_k=3):

    index, chunks = load_knowledge_base()

    retrieved_chunks = retrieve_chunks(
        question,
        index,
        chunks,
        top_k=3
    )

    if not retrieved_chunks:
        return {
            "answer": "No relevant documents were found.",
            "sources": []
        }

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    sources = []

    for item in retrieved_chunks:

        sources.append({
            "chunk_id": item["chunk_id"],
            "source": item["source"],
            "page": item["page"],
            "similarity_score": item["score"]
        })

    return {
        "answer": answer,
        "sources": sources
    }