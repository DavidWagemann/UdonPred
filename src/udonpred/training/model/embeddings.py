"""Attach precomputed per-residue embeddings to a dataset split.

The precomputed-first training path loads embeddings from an ``.h5`` (keyed by
the jsonl ``id``) instead of computing them on the fly. Torch-free: uses only
``datasets`` + ``h5py`` + ``numpy``.
"""


def attach_precomputed_embeddings(
    dataset_split, h5_path, id_column: str = "id", column: str = "embedding_0"
):
    """Add ``column`` to each row from ``h5[str(row[id_column])]``."""
    import h5py
    import numpy as np

    with h5py.File(h5_path, "r") as f:
        table = {key: np.asarray(f[key], dtype=np.float32) for key in f.keys()}

    def _add(row):
        return {column: table[str(row[id_column])].tolist()}

    return dataset_split.map(_add, desc="attaching precomputed embeddings")
