"""Retriever interface. Everything on Day 4 implements this so the eval runner
takes a retriever and does not care what is inside it.

`doc_id` is an optional filter rather than a separate interface: Day 3 found the
corpus-wide and document-scoped configs have different bottlenecks, so every
retriever has to support both scopes for the ablation to test either prediction.
"""

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Hit:
    chunk_id: str
    score: float
    text: str
    metadata: dict = field(default_factory=dict)


class Retriever(Protocol):
    name: str

    def search(self, query: str, k: int, doc_id: str | None = None) -> list[Hit]: ...
