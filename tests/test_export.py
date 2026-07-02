"""Unit tests for the torch-free helpers in :mod:`udonpred.utils.export`.

``export`` defers its ``torch`` import into ``export_checkpoint`` (which needs a
real checkpoint), so the config/checkpoint-discovery helpers below import and
run without the heavy stack installed.
"""

import pytest

# export.py imports PyYAML at module scope (a [training] extra); skip cleanly
# when only the lean core is installed.
pytest.importorskip("yaml")

from udonpred.utils.export import (
    collect_output_keys,
    discover_checkpoints,
    load_config,
    load_hyperparameters,
)


# ---- collect_output_keys -----------------------------------------------------

def test_collect_output_keys_prefers_explicit_outputs():
    config = {
        "config": {"outputs": ["trizod", "disprot"]},
        "data": {"ds": {"fraction": 1.0, "losses": {"k": [{"output": "ignored"}]}}},
    }
    # Explicit config.outputs wins over anything derived from the data section.
    assert collect_output_keys(config) == ["trizod", "disprot"]


def test_collect_output_keys_derived_from_losses_and_metrics():
    config = {
        "config": {},
        "data": {
            "ds1": {
                "fraction": 1.0,
                "losses": {"a": [{"output": "trizod"}]},
                "metrics": {"b": [{"output": "chezod"}]},
            },
            # fraction == 0 datasets are skipped entirely
            "ds2": {"fraction": 0, "losses": {"c": [{"output": "excluded"}]}},
        },
    }
    assert collect_output_keys(config) == ["chezod", "trizod"]  # sorted, deduped


# ---- discover_checkpoints ----------------------------------------------------

def test_discover_checkpoints_requires_both_files(tmp_path):
    good = tmp_path / "run" / "ckpt"
    good.mkdir(parents=True)
    (good / "pytorch_model.bin").write_bytes(b"")
    (good / "config.yaml").write_text("config: {}\n")

    # missing config.yaml -> not a checkpoint
    partial = tmp_path / "run" / "no_config"
    partial.mkdir(parents=True)
    (partial / "pytorch_model.bin").write_bytes(b"")

    found = discover_checkpoints(tmp_path)
    assert found == [good]


# ---- load_config / load_hyperparameters --------------------------------------

def test_load_config_reads_yaml(tmp_path):
    (tmp_path / "config.yaml").write_text("config:\n  input_dim: 1024\n")
    assert load_config(str(tmp_path)) == {"config": {"input_dim": 1024}}


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path))


def test_load_hyperparameters_absent_returns_empty():
    assert load_hyperparameters({"config": {}}) == {}


def test_load_hyperparameters_reads_referenced_file(tmp_path):
    hp = tmp_path / "hp.yaml"
    hp.write_text("learning_rate: 0.001\n")
    config = {"config": {"hyperparameter_path": str(hp)}}
    assert load_hyperparameters(config) == {"learning_rate": 0.001}
