
import pytest

def test_local_embedder():
    from mnemosyne.embeddings.factory import get_embedder
    emb = get_embedder("local", dim=384)
    vecs = emb.embed(["hello world", "test"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384

@pytest.mark.skipif(True, reason="Requires sentence-transformers")
def test_bge_embedder():
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
