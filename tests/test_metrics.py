
from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank
)


def test_precision_at_k():

    retrieved = ["A", "B", "C"]
    relevant = {"A", "C"}

    assert precision_at_k(
        retrieved, relevant, 3
    ) == 2 / 3


def test_recall_at_k():

    retrieved = ["A", "B", "C"]
    relevant = {"A", "C", "D"}

    assert recall_at_k(
        retrieved, relevant, 3
    ) == 2 / 3


def test_reciprocal_rank():

    retrieved = ["A", "B", "C"]
    relevant = {"C"}

    assert reciprocal_rank(
        retrieved, relevant
    ) == 1 / 3


def test_mean_reciprocal_rank():

    retrieved = [
        ["A", "B", "C"],
        ["X", "Y", "A"]
    ]

    relevant = [
        {"A"},
        {"A"}
    ]

    assert mean_reciprocal_rank(
        retrieved, relevant
    ) == 2 / 3
