import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from udonpred.heads import discover_targets

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

EXPECTED = {
    "atlas",
    "chezod",
    "disprot",
    "pdbflex",
    "plddt",
    "softdis",
    "trizod",
    "trizod2",
}


def _run(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-m", "udonpred.caid.predict", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def test_discover_targets_finds_all_onnx_stems(weights_dir):
    assert set(discover_targets(weights_dir)) == EXPECTED


def test_all_targets_writes_one_dir_per_head(tmp_path, weights_dir):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))
    outdir = tmp_path / "out"

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(weights_dir),
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
    # one directory per head, each holding one file per protein
    for target in EXPECTED:
        f = outdir / target / "seq1.caid"
        assert f.exists(), f"missing {f}"
        assert f.read_text().startswith(">seq1\n")


def test_explicit_multiple_targets(tmp_path, weights_dir):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))
    outdir = tmp_path / "out"

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(weights_dir),
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
    assert (outdir / "trizod" / "seq1.caid").exists()
    assert (outdir / "disprot" / "seq1.caid").exists()
    assert not (outdir / "atlas").exists()


def test_each_protein_gets_its_own_file(tmp_path, weights_dir):
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
            str(weights_dir),
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
        # one file per protein, each holding only its own protein
        assert sorted(p.name for p in (outdir / target).iterdir()) == [
            "seq1.caid",
            "seq2.caid",
        ]
        for name in seqs:
            text = (outdir / target / f"{name}.caid").read_text()
            assert text.startswith(f">{name}\n")
            assert text.count(">") == 1
