"""Small local retrievers for tool-search experiments."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .registry import ToolRecord, ToolRegistry, tool_document

_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


@dataclass(frozen=True)
class SearchResult:
    """One retrieved tool and its score."""

    record: ToolRecord
    score: float
    rank: int


class BM25ToolRetriever:
    """Dependency-free BM25 over rendered tool documents."""

    def __init__(self, registry: ToolRegistry, *, k1: float = 1.5, b: float = 0.75):
        self.registry = registry
        self.k1 = k1
        self.b = b
        self._docs = [tool_document(record) for record in registry.records]
        self._term_counts = [Counter(_tokens(doc)) for doc in self._docs]
        self._doc_lens = [sum(counts.values()) for counts in self._term_counts]
        self._avg_len = (
            sum(self._doc_lens) / len(self._doc_lens) if self._doc_lens else 0.0
        )
        df: Counter[str] = Counter()
        for counts in self._term_counts:
            df.update(counts.keys())
        total = max(1, len(self._term_counts))
        self._idf = {
            term: math.log(1 + (total - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def search(self, query: str, *, k: int = 8) -> tuple[SearchResult, ...]:
        terms = _tokens(query)
        scored: list[tuple[float, int]] = []
        for index, counts in enumerate(self._term_counts):
            score = self._score(terms, counts, self._doc_lens[index])
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.registry.records[item[1]].tool_id))
        return tuple(
            SearchResult(
                record=self.registry.records[index],
                score=score,
                rank=rank,
            )
            for rank, (score, index) in enumerate(scored[: max(0, k)], start=1)
        )

    def _score(self, terms: list[str], counts: Counter[str], doc_len: int) -> float:
        if not terms or not counts:
            return 0.0
        score = 0.0
        avg_len = self._avg_len or 1.0
        query_counts = Counter(terms)
        for term, query_tf in query_counts.items():
            tf = counts.get(term, 0)
            if not tf:
                continue
            idf = self._idf.get(term, 0.0)
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / avg_len)
            score += query_tf * idf * (tf * (self.k1 + 1)) / denom
        return score


def _tokens(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]
