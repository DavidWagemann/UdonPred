import numpy as np
import pytest

from udonpred.inference import (
    TARGET_POLICIES,
    binarize_scores,
    normalize_scores,
    unknown_targets,
)


@pytest.mark.parametrize("target", ["trizod", "trizod2", "disprot", "softdis"])
def test_sigmoid_heads_pass_through_unchanged(target):
    scores = np.array([[0.0], [0.25], [1.0]], dtype=np.float32)
    out = normalize_scores(scores, target)
    assert out is scores
    assert TARGET_POLICIES[target].scale is None


def test_chezod_is_flipped_and_rescaled():
    # CheZOD Z-scale -5..16.15, higher = more ordered
    midpoint = (-5.0 + 16.15) / 2
    out = normalize_scores(np.array([-5.0, midpoint, 16.15]), "chezod")
    assert np.allclose(out, [1.0, 0.5, 0.0])


def test_chezod_clamps_out_of_range_values():
    out = normalize_scores(np.array([-12.0, 25.0]), "chezod")
    assert np.allclose(out, [1.0, 0.0])


def test_plddt_is_flipped_and_rescaled():
    out = normalize_scores(np.array([0.0, 50.0, 100.0, 130.0]), "plddt")
    assert np.allclose(out, [1.0, 0.5, 0.0, 0.0])


@pytest.mark.parametrize("target", ["pdbflex", "atlas"])
def test_flexibility_heads_are_rescaled_without_flipping(target):
    # Angstrom-scale 0..10, higher = more flexible/disordered already
    out = normalize_scores(np.array([0.0, 5.0, 10.0, 12.0, -1.0]), target)
    assert np.allclose(out, [0.0, 0.5, 1.0, 1.0, 0.0])


def test_every_head_lands_in_unit_interval():
    raw = np.array([-50.0, -1.0, 0.0, 0.5, 8.0, 16.0, 100.0, 500.0])
    for target, policy in TARGET_POLICIES.items():
        if policy.scale is None:
            continue
        out = normalize_scores(raw, target)
        assert out.min() >= 0.0 and out.max() <= 1.0, target


def test_trailing_dim_is_preserved():
    # format_predictions unwraps a trailing (..., 1) axis, so it must survive
    scores = np.array([[2.0], [8.0], [14.0]])
    out = normalize_scores(scores, "chezod")
    assert out.shape == scores.shape


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="No policy registered"):
        normalize_scores(np.zeros(3), "nonexistent")
    with pytest.raises(ValueError, match="No policy registered"):
        binarize_scores(np.zeros(3), "nonexistent")


def test_unknown_targets_reports_only_unregistered():
    assert unknown_targets(["trizod", "chezod"]) == []
    assert unknown_targets(["trizod", "bogus", "nope"]) == ["bogus", "nope"]


# --- binary column ------------------------------------------------------------


@pytest.mark.parametrize(
    "target,threshold",
    [
        ("trizod", 0.4),
        ("trizod2", 0.4),
        ("disprot", 0.5),
        ("softdis", 0.025),
        ("atlas", 2.0),
        ("pdbflex", 2.0),
    ],
)
def test_disorder_positive_heads_call_at_or_above_threshold(target, threshold):
    raw = np.array([threshold - 0.001, threshold, threshold + 0.001])
    assert binarize_scores(raw, target).tolist() == [0, 1, 1]


@pytest.mark.parametrize("target,threshold", [("chezod", 3.0), ("plddt", 68.8)])
def test_order_positive_heads_call_below_threshold(target, threshold):
    # chezod/plddt run the other way: low raw score means disordered
    raw = np.array([threshold - 0.001, threshold, threshold + 0.001])
    assert binarize_scores(raw, target).tolist() == [1, 0, 0]


def test_binary_calls_are_ints_and_preserve_shape():
    out = binarize_scores(np.array([[0.1], [0.9]]), "trizod")
    assert out.shape == (2, 1)
    assert set(np.unique(out)) <= {0, 1}
    assert np.issubdtype(out.dtype, np.integer)


def test_binary_thresholds_agree_with_normalized_scores():
    # the two columns must not contradict each other: for every head, a residue
    # called disordered must score at least as high as the normalized threshold
    for target, policy in TARGET_POLICIES.items():
        raw = np.linspace(-10.0, 120.0, 400)
        calls = binarize_scores(raw, target)
        norm = normalize_scores(raw, target)
        cutoff = float(normalize_scores(np.array([policy.threshold]), target)[0])
        assert norm[calls == 1].min() >= cutoff - 1e-9, target
        if (calls == 0).any():
            assert norm[calls == 0].max() <= cutoff + 1e-9, target
