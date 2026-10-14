"""Dependency-free retrieval metrics shared by all pilot baselines."""

from __future__ import annotations

from collections.abc import Sequence, Set


def retrieval_metrics(
    rankings: Sequence[Sequence[str]],
    positive_ids: Sequence[Set[str]],
    recall_ks: Sequence[int] = (1, 3),
) -> dict[str, float | int]:
    """Compute first-relevant retrieval metrics from ranked document IDs."""

    if not rankings or len(rankings) != len(positive_ids):
        raise ValueError("rankings and positive_ids must be non-empty and aligned")
    if not recall_ks or any(k < 1 for k in recall_ks):
        raise ValueError("recall cutoffs must be positive")
    ranks: list[int] = []
    for ranking, positives in zip(rankings, positive_ids, strict=True):
        if not positives:
            raise ValueError("every retrieval query requires a positive document")
        rank = next(
            (index for index, document_id in enumerate(ranking, start=1) if document_id in positives),
            None,
        )
        if rank is None:
            raise ValueError("a retrieval ranking is missing its positive document")
        ranks.append(rank)
    result: dict[str, float | int] = {"query_count": len(ranks)}
    for cutoff in recall_ks:
        result[f"recall_at_{cutoff}"] = sum(rank <= cutoff for rank in ranks) / len(ranks)
    result["mrr"] = sum(1.0 / rank for rank in ranks) / len(ranks)
    result["mean_rank"] = sum(ranks) / len(ranks)
    return result


def pairwise_metrics(
    positive_scores: Sequence[float],
    negative_scores: Sequence[Sequence[float]],
) -> dict[str, float | int]:
    """Compare each positive score with each explicit negative score."""

    if not positive_scores or len(positive_scores) != len(negative_scores):
        raise ValueError("positive and negative score groups must be non-empty and aligned")
    margins = [
        float(positive) - float(negative)
        for positive, negatives in zip(positive_scores, negative_scores, strict=True)
        for negative in negatives
    ]
    if not margins:
        raise ValueError("at least one explicit negative score is required")
    return {
        "comparison_count": len(margins),
        "pairwise_accuracy": sum(margin > 0 for margin in margins) / len(margins),
        "mean_similarity_margin": sum(margins) / len(margins),
    }
