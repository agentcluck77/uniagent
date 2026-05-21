from collections.abc import Callable
from typing import Any

import numpy as np


class ToolRAG:
    def __init__(self, tools: list[Callable], backend: str, st_model: str | None):
        self.tools = tools
        self.backend = backend
        self.model: Any = None
        self.vocab: dict[str, int] = {}
        if backend == "tfidf":
            self.matrix = self._fit_tfidf([fn._description for fn in tools])
        elif backend == "sentence-transformers":
            if not st_model:
                raise ValueError("toolrag.st_model is required for sentence-transformers")
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self.model = SentenceTransformer(st_model)
            self.matrix = self._normalise(self.model.encode([fn._description for fn in tools]))
        else:
            raise ValueError(f"unsupported ToolRAG backend: {backend}")

    def retrieve(self, query: str, k: int) -> list[Callable]:
        if k >= len(self.tools):
            return self.tools
        query_vector = self._embed(query)
        scores = self.matrix @ query_vector
        indexes = np.argsort(scores)[::-1][:k]
        return [self.tools[int(index)] for index in indexes]

    def _tokens(self, text: str) -> list[str]:
        tokens = text.lower().replace("/", " ").replace("_", " ").split()
        aliases = {"fetch": ["get", "http"], "webpage": ["http"], "website": ["http"]}
        expanded = list(tokens)
        for token in tokens:
            expanded.extend(aliases.get(token, []))
        return expanded

    def _fit_tfidf(self, docs: list[str]) -> np.ndarray:
        tokenised = [self._tokens(doc) for doc in docs]
        self.vocab = {
            token: index
            for index, token in enumerate(sorted({token for doc in tokenised for token in doc}))
        }
        if not self.vocab:
            return np.zeros((len(docs), 0))
        counts = np.zeros((len(docs), len(self.vocab)))
        for row, doc in enumerate(tokenised):
            for token in doc:
                counts[row, self.vocab[token]] += 1
        doc_freq = (counts > 0).sum(axis=0)
        idf = np.log((1 + len(docs)) / (1 + doc_freq)) + 1
        return self._normalise(counts * idf)

    def _embed(self, text: str) -> np.ndarray:
        if self.backend == "sentence-transformers":
            return self._normalise(self.model.encode([text]))[0]
        vector = np.zeros(len(self.vocab))
        for token in self._tokens(text):
            if token in self.vocab:
                vector[self.vocab[token]] += 1
        return self._normalise(vector)

    def _normalise(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.where(norms == 0, 1, norms)
