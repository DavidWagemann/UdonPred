import numpy as np
import pytest

from caid.embeddings import align_embedding, load_precomputed_embeddings


# ---- align_embedding ---------------------------------------------------------

def test_align_already_trimmed_passthrough():
    emb = np.random.randn(10, 1024).astype(np.float32)
    out = align_embedding(emb, seq_len=10)
    assert out.shape == (10, 1024)
    np.testing.assert_array_equal(out, emb)


def test_align_strips_leading_prefix_token():
    # ProstT5 with <AA2fold> prefix only -> length L+1
    emb = np.random.randn(11, 1024).astype(np.float32)
    out = align_embedding(emb, seq_len=10)
    assert out.shape == (10, 1024)
    np.testing.assert_array_equal(out, emb[1:])


def test_align_strips_prefix_and_trailing_eos():
    # ProstT5 raw last_hidden_state: prefix + residues + eos -> length L+2
    emb = np.random.randn(12, 1024).astype(np.float32)
    out = align_embedding(emb, seq_len=10)
    assert out.shape == (10, 1024)
    np.testing.assert_array_equal(out, emb[1:11])


def test_align_casts_to_float32():
    emb = np.zeros((10, 4), dtype=np.float16)
    out = align_embedding(emb, seq_len=10)
    assert out.dtype == np.float32


def test_align_mismatch_raises():
    with pytest.raises(ValueError):
        align_embedding(np.zeros((8, 4)), seq_len=10)


# ---- load_precomputed_embeddings ---------------------------------------------

def test_load_npy_single_sequence(tmp_path):
    arr = np.random.randn(10, 1024).astype(np.float32)
    np.save(tmp_path / "e.npy", arr)
    src = load_precomputed_embeddings(str(tmp_path / "e.npy"))
    out = src.get("whatever-header", 0)
    assert out.shape == (10, 1024)
    np.testing.assert_array_equal(out, arr)


def test_load_h5_keyed_by_header(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "e.h5", "w") as f:
        f["seq1"] = np.random.randn(5, 1024).astype(np.float32)
        f["seq2"] = np.random.randn(7, 1024).astype(np.float32)
    src = load_precomputed_embeddings(str(tmp_path / "e.h5"))
    assert src.get("seq1", 0).shape == (5, 1024)
    assert src.get("seq2", 1).shape == (7, 1024)


def test_load_h5_missing_header_raises_keyerror(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "e.h5", "w") as f:
        f["seq1"] = np.random.randn(5, 1024).astype(np.float32)
    src = load_precomputed_embeddings(str(tmp_path / "e.h5"))
    with pytest.raises(KeyError):
        src.get("absent", 0)
