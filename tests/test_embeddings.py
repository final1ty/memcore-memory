
import pytest

def test_local_embedder():
    from mnemosyne.embeddings.factory import get_embedder
    from memcore_memory.embeddings.base import BaseEmbedder
    # The shim promises one module instance under both names; a second execution of
    # factory.py under the alias would still hand back a working embedder, so check it.
    import memcore_memory.embeddings.factory as real
    assert get_embedder is real.get_embedder
    emb = get_embedder("local", dim=384)
    assert isinstance(emb, BaseEmbedder)
    vecs = emb.embed(["hello world", "test"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384

def test_bge_embedder():
    # importorskip rather than skipif(True): an image built with
    # EMBEDDING_PROVIDER=bge-small must actually run this.
    pytest.importorskip("sentence_transformers")
    from mnemosyne.embeddings.factory import get_embedder
    emb = get_embedder("bge-small")
    vec = emb.embed_query("hello world")
    assert len(vec) == 384
    # cosine similarity of same query should be high
    import numpy as np
    v1 = emb.embed_query("query: what is python?")
    v2 = emb.embed_query("query: what is python?")
    sim = np.dot(v1, v2)
    assert sim > 0.9
