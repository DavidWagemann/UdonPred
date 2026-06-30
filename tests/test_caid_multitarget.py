import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from caid.predict import discover_targets

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"

EXPECTED = {"atlas", "chezod", "disprot", "pdbflex", "plddt", "softdis", "trizod"}


def _run(args, cwd):
    return subprocess.run(
        [sys.executable, "-m", "caid.predict", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
    )


def test_discover_targets_finds_all_onnx_stems():
    assert set(discover_targets(WEIGHTS)) == EXPECTED


def test_all_targets_writes_one_file_per_head_same_dir(tmp_path):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))
    outdir = tmp_path / "out"

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(WEIGHTS),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--target",
            "all",
            "--output",
            str(outdir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    # one file per head, all in the same flat output directory
    for target in EXPECTED:
        f = outdir / f"{target}.caid"
        assert f.exists(), f"missing {f}"
        assert f.read_text().startswith(">seq1\n")
    # no per-target subdirectories
    assert not any(p.is_dir() for p in outdir.iterdir())


def test_explicit_multiple_targets(tmp_path):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))
    outdir = tmp_path / "out"

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(WEIGHTS),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--target",
            "trizod",
            "disprot",
            "--output",
            str(outdir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    assert (outdir / "trizod.caid").exists()
    assert (outdir / "disprot.caid").exists()
    assert not (outdir / "atlas.caid").exists()


def test_one_file_per_head_holds_all_proteins(tmp_path):
    seqs = {"seq1": "MKTAYIAKQR", "seq2": "GGGCCCDDD"}
    (tmp_path / "in.fasta").write_text(
        "".join(f">{h}\n{s}\n" for h, s in seqs.items())
    )
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "emb.h5", "w") as f:
        for h, s in seqs.items():
            f[h] = np.random.randn(len(s), 1024).astype(np.float32)
    outdir = tmp_path / "out"

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(WEIGHTS),
            "--embeddings",
            str(tmp_path / "emb.h5"),
            "--target",
            "trizod",
            "disprot",
            "--output",
            str(outdir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    for target in ("trizod", "disprot"):
        text = (outdir / f"{target}.caid").read_text()
        # both proteins concatenated into the single per-head file
        assert ">seq1\n" in text and ">seq2\n" in text
        assert text.index(">seq1") < text.index(">seq2")
