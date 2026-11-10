
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank
)


# Example evaluation dataset

evaluation_data = [
    {
        "query": "What is Redis?",
        "relevant_ids": {"chunk_1", "chunk_3"}
    },
    {
        "query": "What is vector embedding?",
        "relevant_ids": {"chunk_4", "chunk_5"}
    },
    {
        "query": "What is FastAPI?",
        "relevant_ids": {"chunk_7"}
    }
]


# Simulated FAISS retrieval results

retrieved_results = [
    ["chunk_1", "chunk_2", "chunk_3"],
    ["chunk_6", "chunk_4", "chunk_5"],
    ["chunk_8", "chunk_7", "chunk_9"]
]


all_relevant = [
    item["relevant_ids"]
    for item in evaluation_data
]

print("RAG EVALUATION RESULTS")
print("----------------------")

for item, retrieved in zip(
        evaluation_data,
        retrieved_results
):

    relevant = item["relevant_ids"]

    precision = precision_at_k(
        retrieved, relevant, 3
    )

    recall = recall_at_k(
        retrieved, relevant, 3
    )

    rr = reciprocal_rank(
        retrieved, relevant
    )

    print(f"\nQuery: {item['query']}")
    print(f"Precision@3: {precision:.2f}")
    print(f"Recall@3: {recall:.2f}")
    print(f"Reciprocal Rank: {rr:.2f}")


mrr = mean_reciprocal_rank(
    retrieved_results,
    all_relevant
)

print("\n----------------------")
print(f"Mean Reciprocal Rank: {mrr:.2f}")
