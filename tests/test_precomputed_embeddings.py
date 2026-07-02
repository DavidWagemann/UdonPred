import pytest

pytest.importorskip("datasets")
pytest.importorskip("h5py")

import h5py
import numpy as np
from datasets import Dataset

from udonpred.training.model.embeddings import attach_precomputed_embeddings


def test_attach_aligns_embeddings_by_id(tmp_path):
    ds = Dataset.from_dict({"id": ["a", "b"], "x_0": ["MK", "AC"]})
    h5 = tmp_path / "e.h5"
    with h5py.File(h5, "w") as f:
        f["a"] = np.arange(6, dtype=np.float32).reshape(3, 2)
        f["b"] = np.ones((2, 2), dtype=np.float32)

    out = attach_precomputed_embeddings(ds, str(h5))
    assert "embedding_0" in out.column_names
    row_a = out[0]
    assert row_a["id"] == "a"
    assert np.allclose(np.array(row_a["embedding_0"]), np.arange(6).reshape(3, 2))
