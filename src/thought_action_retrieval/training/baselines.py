"""Small deterministic TF-IDF and BM25 baselines for the provisional pilot."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any

from thought_action_retrieval.training.metrics import (
    pairwise_metrics,
    retrieval_metrics,
)


FEASIBILITY_LABEL = "FEASIBILITY ONLY — NOT GOLD EVALUATION"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class _SparseScorer:
    def __init__(self, documents: Sequence[str], method: str):
        if method not in {"tfidf", "bm25"}:
            raise ValueError(f"unsupported sparse baseline: {method}")
        if not documents:
            raise ValueError("baseline corpus cannot be empty")
        self.method = method
        self.documents = tuple(documents)
        self.term_counts = tuple(Counter(_tokens(document)) for document in documents)
        self.lengths = tuple(sum(counts.values()) for counts in self.term_counts)
        self.average_length = sum(self.lengths) / len(self.lengths)
        document_frequency = Counter(
            token for counts in self.term_counts for token in counts
        )
        count = len(documents)
        self.tfidf_idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }
        self.bm25_idf = {
            token: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }
        self.tfidf_documents = tuple(self._tfidf_vector(counts) for counts in self.term_counts)

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        vector = {
            token: frequency * self.tfidf_idf[token]
            for token, frequency in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in vector.values())) or 1.0
        return {token: value / norm for token, value in vector.items()}

    def scores(self, query: str) -> list[float]:
        query_counts = Counter(_tokens(query))
        if self.method == "tfidf":
            query_vector = self._tfidf_vector(
                Counter(
                    {
                        token: count
                        for token, count in query_counts.items()
                        if token in self.tfidf_idf
                    }
                )
            )
            return [
                sum(query_vector.get(token, 0.0) * value for token, value in document.items())
                for document in self.tfidf_documents
            ]
        k1 = 1.5
        b = 0.75
        scores: list[float] = []
        for counts, length in zip(self.term_counts, self.lengths, strict=True):
            score = 0.0
            for token in query_counts:
                frequency = counts.get(token, 0)
                if not frequency or token not in self.bm25_idf:
                    continue
                denominator = frequency + k1 * (
                    1 - b + b * length / (self.average_length or 1.0)
                )
                score += self.bm25_idf[token] * frequency * (k1 + 1) / denominator
            scores.append(score)
        return scores


def evaluate_records(
    records: Sequence[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    """Evaluate a sparse method over all unique Actions in a split."""

    if not records:
        raise ValueError("baseline records cannot be empty")
    documents = sorted(
        {
            row["positive_action"]["serialized"]
            for row in records
        }
        | {
            negative["serialized"]
            for row in records
            for negative in row.get("negative_actions", [])
        }
    )
    scorer = _SparseScorer(documents, method)
    rankings: list[list[str]] = []
    positives: list[set[str]] = []
    positive_scores: list[float] = []
    negative_scores: list[list[float]] = []
    document_index = {document: index for index, document in enumerate(documents)}
    for row in records:
        scores = scorer.scores(row["intent_input"])
        ranking = [
            document
            for _, document in sorted(
                zip(scores, documents, strict=True),
                key=lambda item: (-item[0], item[1]),
            )
        ]
        positive = row["positive_action"]["serialized"]
        rankings.append(ranking)
        positives.append({positive})
        positive_scores.append(scores[document_index[positive]])
        negative_scores.append(
            [scores[document_index[item["serialized"]]] for item in row.get("negative_actions", [])]
        )
    return {
        "label": FEASIBILITY_LABEL,
        "method": method,
        "retrieval": retrieval_metrics(rankings, positives),
        "hard_negative": pairwise_metrics(positive_scores, negative_scores),
        "document_count": len(documents),
    }
