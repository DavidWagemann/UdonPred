import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def _run(args, cwd):
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    return subprocess.run(
        [sys.executable, "-m", "udonpred.caid.predict", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )


def test_caid_predict_npy_single_sequence_to_stdout(tmp_path, weights_dir):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    # L+2 layout (prefix + residues + eos) to also exercise alignment
    emb = np.random.randn(len(seq) + 2, 1024).astype(np.float32)
    np.save(tmp_path / "emb.npy", emb)

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(weights_dir),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--target",
            "trizod",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    lines = [l for l in res.stdout.splitlines() if l.strip()]
    assert lines[0] == ">seq1"
    # one row per residue
    assert len(lines) == 1 + len(seq)
    # rows: idx \t residue \t score
    first = lines[1].split("\t")
    assert first[0] == "1" and first[1] == "M"
    assert 0.0 <= float(first[2]) <= 1.0


def test_caid_predict_writes_one_file_per_head(tmp_path, weights_dir):
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
            "--output",
            str(outdir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    # file is named after the prediction head, not the sequence
    out_file = outdir / "udonpred_trizod.caid"
    assert out_file.exists()
    assert out_file.read_text().startswith(">seq1\n")


def _scores(path):
    return [
        float(line.split("\t")[2])
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    ]


def _run_target(tmp_path, weights_dir, target, outdir, *extra):
    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(weights_dir),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--target",
            target,
            "--output",
            str(outdir),
            "--device",
            "cpu",
            *extra,
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    return _scores(outdir / f"udonpred_{target}.caid")


def test_normalize_bounds_and_flips_chezod(tmp_path, weights_dir):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))

    normalized = _run_target(tmp_path, weights_dir, "chezod", tmp_path / "norm")
    raw = _run_target(
        tmp_path, weights_dir, "chezod", tmp_path / "raw", "--no-normalize"
    )

    assert len(normalized) == len(raw) == len(seq)
    # CAID requires [0, 1]; the raw CheZOD head is an unbounded regression
    assert all(0.0 <= s <= 1.0 for s in normalized)
    # chezod is order-positive, so normalizing must invert the ranking
    assert np.corrcoef(normalized, raw)[0, 1] < 0


def test_normalize_leaves_sigmoid_head_untouched(tmp_path, weights_dir):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))

    normalized = _run_target(tmp_path, weights_dir, "trizod", tmp_path / "norm")
    raw = _run_target(
        tmp_path, weights_dir, "trizod", tmp_path / "raw", "--no-normalize"
    )
    assert normalized == raw


def test_normalize_rejects_head_without_policy(tmp_path, weights_dir):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))
    fake_dir = tmp_path / "heads"
    fake_dir.mkdir()
    (fake_dir / "bogus.onnx").write_bytes(b"")

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(fake_dir),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--target",
            "bogus",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode != 0
    # fails on the policy check, before trying to load the (invalid) head
    assert "--normalize has no policy for: bogus" in res.stderr
