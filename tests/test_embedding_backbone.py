"""Unit tests for :mod:`udonpred.embedding.backbone`.

These exercise the parts of the ProstT5 backbone helper that don't require a
downloaded model. They still need ``torch`` (the ``udonpred[embedding]`` extra),
so the whole module is skipped when torch is unavailable.
"""

import pytest

pytest.importorskip("torch")

from udonpred.embedding.backbone import resolve_device


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_auto_is_concrete():
    assert resolve_device("auto") in ("cpu", "cuda")


def test_resolve_device_rejects_unknown():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_on_the_fly_predictor_module_imports():
    # Smoke test: the on-the-fly runner wires up against the relocated backbone.
    import udonpred.embedding.predict as predict

    assert hasattr(predict, "run_exported")
    assert hasattr(predict, "main")
