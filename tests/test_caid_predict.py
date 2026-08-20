import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

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
    # rows: idx \t residue \t score \t binary
    first = lines[1].split("\t")
    assert len(first) == 4
    assert first[0] == "1" and first[1] == "M"
    assert 0.0 <= float(first[2]) <= 1.0
    assert first[3] in ("0", "1")


def test_caid_predict_writes_one_dir_per_head_one_file_per_protein(
    tmp_path, weights_dir
):
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
    # CAID layout: {target}/{protein}.caid
    out_file = outdir / "trizod" / "seq1.caid"
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
    return _scores(outdir / target / "seq1.caid")


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
    assert "No policy registered for: bogus" in res.stderr


def _rows(path):
    return [
        line.split("\t")
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith(">")
    ]


def test_binary_column_matches_raw_threshold(tmp_path, weights_dir):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQ"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))

    for target, threshold, order_positive in [
        ("trizod", 0.4, False),
        ("chezod", 3.0, True),
        ("plddt", 68.8, True),
        ("atlas", 2.0, False),
    ]:
        outdir = tmp_path / f"raw_{target}"
        _run_target(tmp_path, weights_dir, target, outdir, "--no-normalize")
        rows = _rows(outdir / target / "seq1.caid")
        assert len(rows) == len(seq)
        checked = 0
        for row in rows:
            assert len(row) == 4, row
            raw, call = float(row[2]), row[3]
            # the score column is rounded to 3 decimals, so a raw value within
            # half a step of the cutoff can land on either side of it
            if abs(raw - threshold) <= 5e-4:
                continue
            expected = raw < threshold if order_positive else raw >= threshold
            assert call == str(int(expected)), (target, row)
            checked += 1
        assert checked, f"no unambiguous rows to check for {target}"


def test_binary_column_is_unaffected_by_normalization(tmp_path, weights_dir):
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))

    _run_target(tmp_path, weights_dir, "chezod", tmp_path / "norm")
    _run_target(tmp_path, weights_dir, "chezod", tmp_path / "raw", "--no-normalize")

    normalized = [r[3] for r in _rows(tmp_path / "norm" / "chezod" / "seq1.caid")]
    raw = [r[3] for r in _rows(tmp_path / "raw" / "chezod" / "seq1.caid")]
    assert normalized == raw
    # a normalized disorder call must sit at the high end of the score column
    scores = [float(r[2]) for r in _rows(tmp_path / "norm" / "chezod" / "seq1.caid")]
    disordered = [s for s, c in zip(scores, normalized) if c == "1"]
    ordered = [s for s, c in zip(scores, normalized) if c == "0"]
    if disordered and ordered:
        # 1e-3 tolerance: the score column is rounded to 3 decimals
        assert min(disordered) >= max(ordered) - 1e-3


def test_timings_csv_written_next_to_each_flavors_predictions(tmp_path, weights_dir):
    seqs = {"P04637": "MKTAYIAKQRQ", "P38398": "GGGCCCDDD"}
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
            "chezod",
            "--output",
            str(outdir),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr

    for target in ("trizod", "chezod"):
        lines = (outdir / target / "timings.csv").read_text().splitlines()
        assert lines[0] == "# Running UdonPred"
        assert lines[1] == "sequence,milliseconds"
        # one row per input protein, in input order, alongside its .caid file
        assert [line.split(",")[0] for line in lines[2:]] == list(seqs)
        for line in lines[2:]:
            assert int(line.split(",")[1]) >= 0
            assert (outdir / target / f"{line.split(',')[0]}.caid").exists()


def test_no_timings_csv_when_writing_to_stdout(tmp_path, weights_dir):
    seq = "MKTAYIAKQR"
    (tmp_path / "in.fasta").write_text(f">seq1\n{seq}\n")
    np.save(tmp_path / "emb.npy", np.random.randn(len(seq), 1024).astype(np.float32))

    res = _run(
        [
            str(tmp_path / "in.fasta"),
            str(weights_dir),
            "--embeddings",
            str(tmp_path / "emb.npy"),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    assert res.returncode == 0, res.stderr
    assert "timings" not in res.stdout
    assert not list(tmp_path.glob("**/timings.csv"))
