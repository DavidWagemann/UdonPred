"""The on-the-fly runner's output layout, verified without a real backbone.

``udonpred.embedding.predict`` imports torch at module scope and would otherwise
need the ProstT5 weights, so torch is stubbed and the backbone helpers are
replaced with random embeddings. The ONNX heads are real, so this exercises the
actual scoring, post-processing, and writing path — the part shared with the
CAID runner.
"""

import contextlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def predict_module(monkeypatch):
    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")
        torch_stub.float16 = "float16"
        torch_stub.float32 = "float32"
        torch_stub.inference_mode = contextlib.nullcontext
        monkeypatch.setitem(sys.modules, "torch", torch_stub)

    import udonpred.embedding.predict as predict

    monkeypatch.setattr(predict, "resolve_device", lambda device: "cpu")
    monkeypatch.setattr(predict, "load_tokenizer", lambda: object())
    monkeypatch.setattr(predict, "load_backbone", lambda device, dtype: object())
    monkeypatch.setattr(
        predict, "tokenize_batch", lambda tok, seqs, device: (None, None)
    )
    return predict


def _fake_embeddings(seqs):
    max_len = max(len(s) for s in seqs)
    return np.random.randn(len(seqs), max_len, 1024).astype(np.float32)


def test_on_the_fly_runner_writes_the_caid_layout(
    tmp_path, weights_dir, predict_module, monkeypatch
):
    entries = [("P04637", "MKTAYIAKQRQ"), ("sp|P38398|X", "GGGCCCDDD")]
    seqs = [s for _, s in entries]
    monkeypatch.setattr(
        predict_module,
        "compute_embeddings",
        lambda *a, **k: _fake_embeddings(seqs),
    )

    predict_module.run_exported(
        entries,
        str(weights_dir),
        ["trizod", "chezod"],
        max_total_seq_len=2000,
        output_path=str(tmp_path / "out"),
        device="cpu",
        smooth=1.5,
    )

    out = tmp_path / "out"
    # same layout as the CAID runner: {target}/{protein}.caid
    for target in ("trizod", "chezod"):
        assert sorted(p.name for p in (out / target).glob("*.caid")) == [
            "P04637.caid",
            "sp_P38398_X.caid",
        ]
    text = (out / "trizod" / "P04637.caid").read_text()
    assert text.startswith(">P04637\n")
    rows = [line.split("\t") for line in text.splitlines()[1:]]
    assert len(rows) == len("MKTAYIAKQRQ")
    for row in rows:
        # four columns, three-decimal score, binary call
        assert len(row) == 4
        assert len(row[2].split(".")[1]) == 3
        assert 0.0 <= float(row[2]) <= 1.0
        assert row[3] in ("0", "1")


def test_on_the_fly_runner_rejects_head_without_policy(
    tmp_path, weights_dir, predict_module
):
    fake_dir = tmp_path / "heads"
    fake_dir.mkdir()
    (fake_dir / "bogus.onnx").write_bytes(b"")
    with pytest.raises(ValueError, match="No policy registered"):
        predict_module.run_exported(
            [("seq1", "MK")],
            str(fake_dir),
            ["bogus"],
            max_total_seq_len=2000,
            output_path=str(tmp_path / "out"),
            device="cpu",
        )


def test_on_the_fly_runner_writes_timings_per_flavor(
    tmp_path, weights_dir, predict_module, monkeypatch
):
    entries = [("P04637", "MKTAYIAKQRQ"), ("P38398", "GGGCCCDDD")]
    seqs = [s for _, s in entries]
    monkeypatch.setattr(
        predict_module,
        "compute_embeddings",
        lambda *a, **k: _fake_embeddings(seqs),
    )

    predict_module.run_exported(
        entries,
        str(weights_dir),
        ["trizod", "plddt"],
        max_total_seq_len=2000,
        output_path=str(tmp_path / "out"),
        device="cpu",
        smooth=1.5,
    )

    for target in ("trizod", "plddt"):
        lines = (tmp_path / "out" / target / "timings.csv").read_text().splitlines()
        assert lines[0] == "# Running UdonPred"
        assert lines[1] == "sequence,milliseconds"
        assert [line.split(",")[0] for line in lines[2:]] == ["P04637", "P38398"]
        for line in lines[2:]:
            assert int(line.split(",")[1]) >= 0
