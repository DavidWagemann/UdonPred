import numpy as np
import pytest

from udonpred.inference import (
    TARGET_NORMALIZATION,
    normalize_scores,
    unknown_normalization_targets,
)


@pytest.mark.parametrize("target", ["trizod", "trizod2", "disprot", "softdis"])
def test_sigmoid_heads_pass_through_unchanged(target):
    scores = np.array([[0.0], [0.25], [1.0]], dtype=np.float32)
    out = normalize_scores(scores, target)
    assert out is scores
    assert TARGET_NORMALIZATION[target] is None


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
    for target, policy in TARGET_NORMALIZATION.items():
        if policy is None:
            continue
        out = normalize_scores(raw, target)
        assert out.min() >= 0.0 and out.max() <= 1.0, target


def test_trailing_dim_is_preserved():
    # format_predictions unwraps a trailing (..., 1) axis, so it must survive
    scores = np.array([[2.0], [8.0], [14.0]])
    out = normalize_scores(scores, "chezod")
    assert out.shape == scores.shape


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="No normalization policy"):
        normalize_scores(np.zeros(3), "nonexistent")


def test_unknown_normalization_targets_reports_only_unregistered():
    assert unknown_normalization_targets(["trizod", "chezod"]) == []
    assert unknown_normalization_targets(["trizod", "bogus", "nope"]) == ["bogus", "nope"]
