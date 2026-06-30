from pathlib import Path

import numpy as np
import pytest

from udonpred.inference import (
    count_batches,
    iter_batches,
    load_head,
    score_embeddings,
    smooth_scores,
)

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"


def test_smooth_scores_preserves_length_and_disabled_passthrough():
    scores = np.array([0.0, 1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    smoothed = smooth_scores(scores, sigma=1.5)
    assert smoothed.shape == scores.shape
    # smoothing pulls the spikes toward the mean
    assert smoothed[1] < 1.0 and smoothed[2] > 0.0
    # sigma <= 0 is a no-op
    np.testing.assert_array_equal(smooth_scores(scores, sigma=0), scores)


def test_iter_and_count_batches_agree():
    items = [("a", "X" * 1500), ("b", "X" * 600), ("c", "X" * 600)]
    batches = list(iter_batches(items, max_total_len=2000))
    assert count_batches(items, 2000) == len(batches)
    # every entry appears exactly once across batches
    flat = [h for batch in batches for h, _ in batch]
    assert flat == ["a", "b", "c"]


def test_score_embeddings_real_head_shape_and_range():
    head = load_head(WEIGHTS / "trizod.onnx", "cpu")
    emb = np.random.randn(1, 10, 1024).astype(np.float32)
    scores = score_embeddings(head, emb)
    # one batch item, 10 residues
    assert scores.shape[0] == 1
    assert scores.shape[1] == 10
    flat = scores.reshape(-1)
    # sigmoid output -> probabilities in [0, 1]
    assert flat.min() >= 0.0 and flat.max() <= 1.0
