"""Loading and aligning precomputed ProstT5 embeddings for the CAID runner.

Embeddings are supplied by the CAID organizers (we never run the PLM in the
container). Two container formats are supported:

* ``.npy`` — a single ``(L, D)`` array for a single-sequence FASTA.
* ``.h5``  — one dataset per sequence, keyed by the FASTA header.

Per-residue embeddings may or may not include the ProstT5 ``<AA2fold>`` prefix
token and trailing ``</s>`` (EOS). :func:`align_embedding` trims these so the
result has exactly one row per residue.
"""

from pathlib import Path
from typing import Dict

import numpy as np


def align_embedding(
    embedding: np.ndarray,
    seq_len: int,
    prefix_len: int = 1,
) -> np.ndarray:
    """Trim a per-token embedding down to exactly ``seq_len`` residue rows.

    Accepts the three layouts the organizers might provide and returns a
    ``float32`` array of shape ``(seq_len, D)``:

    * ``L``      — already trimmed (used as-is).
    * ``L + prefix_len``     — leading ``<AA2fold>`` prefix only (prefix stripped).
    * ``L + prefix_len + 1`` — prefix + trailing EOS (both stripped).

    Any other length is a genuine mismatch and raises ``ValueError``.
    """
    emb = np.asarray(embedding, dtype=np.float32)
    n_tokens = emb.shape[0]

    if n_tokens == seq_len:
        aligned = emb
    elif n_tokens == seq_len + prefix_len:
        aligned = emb[prefix_len:]
    elif n_tokens == seq_len + prefix_len + 1:
        aligned = emb[prefix_len : prefix_len + seq_len]
    else:
        raise ValueError(
            f"Embedding length {n_tokens} is incompatible with sequence "
            f"length {seq_len} (expected {seq_len}, {seq_len + prefix_len}, "
            f"or {seq_len + prefix_len + 1} rows for prefix_len={prefix_len})."
        )
    return aligned


class _NpyEmbeddingSource:
    """A single ``(L, D)`` embedding loaded from a ``.npy`` file."""

    def __init__(self, array: np.ndarray):
        self._array = np.asarray(array)

    def get(self, header: str, index: int) -> np.ndarray:
        if index != 0:
            raise IndexError(
                "A .npy embedding file holds a single sequence; received "
                f"request for sequence index {index} (header {header!r}). "
                "Use an .h5 file for multi-sequence inputs."
            )
        return self._array


class _H5EmbeddingSource:
    """Per-sequence embeddings loaded from an ``.h5`` file, keyed by header."""

    def __init__(self, mapping: Dict[str, np.ndarray]):
        self._mapping = mapping

    def get(self, header: str, index: int) -> np.ndarray:
        if header not in self._mapping:
            raise KeyError(
                f"No embedding found for sequence header {header!r} in the "
                f".h5 file. Available keys: {sorted(self._mapping)[:5]}..."
            )
        return self._mapping[header]


def load_precomputed_embeddings(path: str):
    """Load an embedding file, dispatching on extension (``.npy`` or ``.h5``).

    Returns an object exposing ``get(header, index) -> np.ndarray``.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        return _NpyEmbeddingSource(np.load(path))
    if suffix in (".h5", ".hdf5"):
        import h5py

        mapping: Dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                mapping[key] = f[key][:]
        return _H5EmbeddingSource(mapping)
    raise ValueError(
        f"Unsupported embedding format {suffix!r}; expected .npy or .h5."
    )
