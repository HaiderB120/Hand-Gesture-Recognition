# tests/test_matching_templates.py
import importlib
import numpy as np

def test_build_templates_and_cosine(sample_gesture_db, monkeypatch):
    gr = importlib.import_module("single_gesture_run")

    # Monkeypatch load_db to return our sample db
    monkeypatch.setattr(gr, "load_db", lambda: sample_gesture_db, raising=True)
    templates = gr.build_templates(sample_gesture_db)
    assert "peace" in templates and "copy_gesture" in templates

    # Means should be L2-normalized
    for t in templates.values():
        mean = t["mean"]
        n = np.linalg.norm(mean)
        assert 0.99 <= n <= 1.01

    # Cosine similarity sanity: a vector compared to itself is ~1.0
    v = np.array(sample_gesture_db["peace"]["samples"][0], dtype=np.float32)
    v = v / np.linalg.norm(v)
    sim = gr.cosine_sim(v, templates["peace"]["mean"])
    assert sim > 0.99
