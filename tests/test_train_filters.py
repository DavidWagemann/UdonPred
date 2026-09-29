"""Experiment filters on the training data (needs the ``training`` stack)."""

import pytest

datasets = pytest.importorskip("datasets")
data = pytest.importorskip("udonpred.training.model.data")


def _split(lengths):
    return datasets.Dataset.from_dict({
        "id": [f"p{i}" for i in range(len(lengths))],
        "x_0": ["M" * n for n in lengths],
        "y": [[0.5] * n for n in lengths],
    })


@pytest.fixture
def cached_dataset(tmp_path, monkeypatch):
    """A built dataset cached where get_datasets looks for it, so nothing is embedded."""
    monkeypatch.setattr(data, "Embedder", lambda **kwargs: None)
    path = tmp_path / "trizod"
    datasets.DatasetDict({
        "train": _split([10, 24, 25, 60]),
        "validation": _split([10, 60]),
        "test": _split([10]),
    }).map(lambda row: {"dataset": "trizod"}).save_to_disk(str(path / "hf-prostt5"))

    def config(**settings):
        return {
            "data": {"trizod": {"path": str(path), "fraction": 1}},
            "config": {
                "backbone": {"name": "Rostlab/ProstT5_fp16", "prefix_token": "<AA2fold>",
                             "tokenizer_type": "T5Tokenizer", "model_type": "T5EncoderModel"},
                "embeddings": {"plm": "prostt5"},
                "min_length": 0,
                **settings,
            },
        }

    return config


def _lengths(ds, split):
    return sorted(len(seq) for seq in ds[split]["x_0"])


def test_train_min_length_filters_only_the_train_split(cached_dataset):
    ds = data.get_datasets(cached_dataset(train_min_length=25))
    assert _lengths(ds, "train") == [25, 60]
    # valid/test stay complete, so metrics compare against the baseline
    assert _lengths(ds, "validation") == [10, 60]
    assert _lengths(ds, "test") == [10]


def test_train_min_length_off_keeps_everything(cached_dataset):
    ds = data.get_datasets(cached_dataset(train_min_length=0))
    assert _lengths(ds, "train") == [10, 24, 25, 60]
