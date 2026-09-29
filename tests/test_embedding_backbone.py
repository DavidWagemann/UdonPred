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


def test_embedding_runner_uses_the_shared_cli_and_output_contract():
    """The runner must expose the shared flags and write through the shared writer."""
    from udonpred.embedding.predict import build_parser as embedding_parser

    def flags(parser):
        return {
            opt
            for action in parser._actions
            for opt in action.option_strings
            if opt.startswith("--")
        }

    for flag in ("--target", "--output", "--smooth", "--normalize", "--revision"):
        assert flag in flags(embedding_parser()), flag

    # the on-the-fly runner writes through the same shared writer
    import udonpred.embedding.predict as predict
    from udonpred.output import CaidWriter

    assert predict.CaidWriter is CaidWriter
