
def precision_at_k(retrieved_ids, relevant_ids, k):

    if k <= 0:
        raise ValueError("k must be greater than zero")

    retrieved = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / k


def recall_at_k(retrieved_ids, relevant_ids, k):

    if not relevant_ids:
        return 0.0

    retrieved = retrieved_ids[:k]

    relevant_retrieved = sum(
        1 for chunk_id in retrieved
        if chunk_id in relevant_ids
    )

    return relevant_retrieved / len(relevant_ids)


def reciprocal_rank(retrieved_ids, relevant_ids):

    for rank, chunk_id in enumerate(retrieved_ids, start=1):

        if chunk_id in relevant_ids:
            return 1 / rank

    return 0.0


def mean_reciprocal_rank(all_retrieved, all_relevant):

    if not all_retrieved:
        return 0.0

    reciprocal_ranks = []

    for retrieved, relevant in zip(
            all_retrieved,
            all_relevant
    ):

        reciprocal_ranks.append(
            reciprocal_rank(retrieved, relevant)
        )

    return sum(reciprocal_ranks) / len(reciprocal_ranks)
