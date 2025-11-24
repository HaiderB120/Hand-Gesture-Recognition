# src/core/gesture_db.py

import json
import os
from typing import Dict, Any

import numpy as np

DATA_PATH = "gestures.json"


def load_db() -> Dict[str, Any]:
    """Load gestures.json if it exists; otherwise raise FileNotFoundError."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"{DATA_PATH} not found. Run v1_gesture_train.py first.")
    with open(DATA_PATH, "r") as f:
        return json.load(f)


def save_db(db: Dict[str, Any]) -> None:
    """Write the in-memory gesture database back to gestures.json (human-readable)."""
    with open(DATA_PATH, "w") as f:
        json.dump(db, f, indent=2)


def build_templates(db: Dict[str, Any]):
    """
    Average all samples for each gesture to make a single template vector.
    Also L2-normalize the mean vector for stable cosine similarity.
    Returns: { name: {'action': str, 'mean': np.ndarray}, ... }
    """
    templates = {}
    for name, item in db.items():
        samples = np.array(item.get("samples", []), dtype=np.float32)
        if len(samples) == 0:
            continue

        mean_vec = samples.mean(axis=0)
        n = np.linalg.norm(mean_vec)
        if n > 1e-6:
            mean_vec = mean_vec / n

        templates[name] = {
            "action": item.get("action", "screenshot"),
            "mean": mean_vec,
        }
    return templates
