"""The per-pLM dataset cache under data/split/<target>/ (needs the ``training`` stack)."""

import pytest

datasets = pytest.importorskip("datasets")
data = pytest.importorskip("udonpred.training.model.data")


def _split(n):
    return datasets.Dataset.from_dict(
        {"id": [f"p{i}" for i in range(n)], "x_0": ["MK"] * n, "y": [[0.5, 0.5]] * n, "dataset": ["trizod"] * n}
    )


def test_a_built_dataset_is_cached_only_once_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "Embedder", lambda **kwargs: None)
    built = datasets.DatasetDict({"train": _split(3), "validation": _split(2), "test": _split(1)})
    monkeypatch.setattr(data, "build_datasets", lambda *args, **kwargs: built)
    saved_to = []
    save = built.save_to_disk
    monkeypatch.setattr(built, "save_to_disk", lambda path, **kw: (saved_to.append(path), save(path, **kw)))
    target = tmp_path / "trizod"
    (target / "hf-prostt5.partial").mkdir(parents=True)  # left behind by a killed run

    data.get_datasets({
        "data": {"trizod": {"path": str(target), "fraction": 1}},
        "config": {"backbone": {"name": "b", "prefix_token": "", "tokenizer_type": "t", "model_type": "m"},
                   "embeddings": {"plm": "prostt5"}, "min_length": 0},
    })

    # written under a temporary name, then renamed: a kill mid-save leaves no loadable cache
    assert saved_to == [str(target / "hf-prostt5.partial")]
    assert (target / "hf-prostt5" / "dataset_dict.json").exists()
    assert not (target / "hf-prostt5.partial").exists()
