import math
from pathlib import Path
import tempfile
from unittest.mock import MagicMock
import pytest

from memory.db import DatabaseManager
from memory.embeddings import EmbeddingManager, cosine_similarity
from memory.store import MemoryStore


def test_cosine_similarity_math():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v2), 1.0, rel_tol=1e-5)

    v_orth = [0.0, 1.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v_orth), 0.0, rel_tol=1e-5)

    v_opp = [-1.0, 0.0, 0.0]
    assert math.isclose(cosine_similarity(v1, v_opp), -1.0, rel_tol=1e-5)

    # Empty or zero vectors
    assert cosine_similarity([], []) == 0.0
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_vector_blob_persistence_and_cascade_delete():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "emb_test.db"
        mgr = DatabaseManager(db_path=db_path)
        emb_mgr = EmbeddingManager(dimension=4, db_manager=mgr)

        # 1. Insert parent memory first
        with mgr.connection() as conn:
            conn.execute(
                """
                INSERT INTO memories (id, content, created_at, updated_at, last_accessed)
                VALUES ('mem_test_emb', 'Test memory content', '2026-01-01', '2026-01-01', '2026-01-01');
                """
            )

        # 2. Store vector
        test_vector = [0.123, -0.456, 0.789, 1.0]
        emb_mgr.store_embedding("mem_test_emb", test_vector)

        # 3. Retrieve and verify
        retrieved = emb_mgr.get_embedding("mem_test_emb")
        assert retrieved is not None
        assert len(retrieved) == 4
        for a, b in zip(test_vector, retrieved):
            assert math.isclose(a, b, rel_tol=1e-5)

        # 4. Delete parent memory -> CASCADE deletes embedding
        with mgr.connection() as conn:
            conn.execute("DELETE FROM memories WHERE id = 'mem_test_emb';")

        assert emb_mgr.get_embedding("mem_test_emb") is None


def test_search_similar_ranking():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "search_test.db"
        mgr = DatabaseManager(db_path=db_path)
        emb_mgr = EmbeddingManager(dimension=3, db_manager=mgr)

        with mgr.connection() as conn:
            conn.execute("INSERT INTO memories (id, content, status, created_at, updated_at, last_accessed) VALUES ('m1', 'doc1', 'active', 'now', 'now', 'now');")
            conn.execute("INSERT INTO memories (id, content, status, created_at, updated_at, last_accessed) VALUES ('m2', 'doc2', 'active', 'now', 'now', 'now');")
            conn.execute("INSERT INTO memories (id, content, status, created_at, updated_at, last_accessed) VALUES ('m3', 'doc3', 'superseded', 'now', 'now', 'now');")

        # m1 is closest to [1, 0, 0]
        emb_mgr.store_embedding("m1", [0.9, 0.1, 0.0])
        # m2 is farther
        emb_mgr.store_embedding("m2", [0.1, 0.9, 0.0])
        # m3 is inactive (superseded)
        emb_mgr.store_embedding("m3", [1.0, 0.0, 0.0])

        hits = emb_mgr.search_similar([1.0, 0.0, 0.0], top_k=5)
        # m3 should NOT be returned because status != 'active'
        hit_ids = [h[0] for h in hits]
        assert "m3" not in hit_ids
        assert hit_ids[0] == "m1"
        assert hit_ids[1] == "m2"
        assert hits[0][1] > hits[1][1]


def test_reciprocal_rank_fusion_math():
    fts_ranks = ["docA", "docB", "docC"]
    vec_ranks = ["docB", "docA", "docD"]

    fused = EmbeddingManager.calculate_rrf(fts_ranks=fts_ranks, vec_ranks=vec_ranks, k=60)
    assert len(fused) == 4
    top_ids = [doc_id for doc_id, _ in fused]
    # docA and docB appear in both lists and should occupy the top 2 slots
    assert set(top_ids[:2]) == {"docA", "docB"}
    assert "docD" in top_ids
    assert "docC" in top_ids


def test_hybrid_recall_with_fallback():
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "hybrid_test.db"
        store = MemoryStore(db_path=db_path)

        # Store two facts
        e1 = store.remember("Hyprland blur configuration in ~/.config/hypr/hyprland.conf", tags=["hyprland", "blur"])
        e2 = store.remember("VSCode extension settings in ~/.config/Code/User/settings.json", tags=["vscode"])

        # 1. Fallback when embeddings returns None (pure FTS5)
        store.embeddings.embed_text = MagicMock(return_value=None)
        results = store.recall("blur", hybrid=True)
        assert len(results) >= 1
        assert results[0].id == e1.id

        # 2. Hybrid search where vector search returns match
        mock_vec = [0.1] * 768
        store.embeddings.embed_text = MagicMock(return_value=mock_vec)
        store.embeddings.search_similar = MagicMock(return_value=[(e2.id, 0.95), (e1.id, 0.4)])

        hybrid_results = store.recall("settings", hybrid=True)
        assert len(hybrid_results) >= 1
        assert hybrid_results[0].id == e2.id


def test_live_ollama_embedding_if_available():
    mgr = EmbeddingManager()
    if not mgr.is_available():
        pytest.skip("Local Ollama embedding model not available in current environment")

    vector = mgr.embed_text("Testing J.A.R.V.I.S. semantic memory embeddings")
    assert vector is not None
    assert len(vector) == 768
    assert any(x != 0.0 for x in vector)
