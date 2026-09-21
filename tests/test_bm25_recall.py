"""Regression test for the BM25 recall hole.

BM25Okapi's IDF, log((N-df+0.5)/(df+0.5)), reaches zero when a term appears in half
the documents and goes negative past that. Because the retriever drops non-positive
scores, the terms that recur most in a personal memory store - exactly the ones worth
searching for - returned nothing at all. Switching to BM25L keeps non-matching
documents at zero while scoring every real match positive.
"""

import pytest

from memcore_memory.retrieval.retrievers.bm25 import BM25Retriever


class _Item:
    def __init__(self, id, content):
        self.id, self.content = id, content


class _Store:
    def __init__(self, contents):
        self.items = [_Item(str(i), c) for i, c in enumerate(contents)]

    async def list_all(self):
        return self.items


CORPUS = [
    "Docker containers run on SkyNAS",
    "The cat sat on the mat",
    "Postgres is a relational database",
    "SkyNAS uses Ubuntu as its operating system",
]


@pytest.mark.parametrize("query,expected_hits", [
    ("Docker", 1),
    ("SkyNAS", 2),      # in half the corpus - returned 0 under BM25Okapi
    ("on", 2),          # also in half the corpus
    ("cat", 1),
    ("database", 1),
    ("nonexistentword", 0),
])
async def test_terms_are_retrievable_regardless_of_document_frequency(query, expected_hits):
    results = await BM25Retriever(_Store(CORPUS)).retrieve(query, k=10)
    assert len(results) == expected_hits


async def test_term_in_every_document_still_retrieves():
    store = _Store(["shared token alpha", "shared token beta", "shared token gamma"])
    results = await BM25Retriever(store).retrieve("shared", k=10)
    assert len(results) == 3


async def test_non_matching_documents_are_excluded():
    """BM25Plus would have scored all four positive; precision must survive the fix."""
    results = await BM25Retriever(_Store(CORPUS)).retrieve("Docker", k=10)
    assert [r["id"] for r in results] == ["0"]


async def test_index_is_rebuilt_when_the_corpus_changes():
    store = _Store(["first document"])
    retriever = BM25Retriever(store)
    assert len(await retriever.retrieve("first", k=5)) == 1
    store.items.append(_Item("1", "second document about first things"))
    assert len(await retriever.retrieve("first", k=5)) == 2


async def test_empty_store_returns_nothing():
    assert await BM25Retriever(_Store([])).retrieve("anything", k=5) == []
