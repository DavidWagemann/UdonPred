"""Unit tests for :mod:`udonpred.training.run`.

Importing the training entry point pulls in the heavy ``udonpred[training]``
stack (transformers, wandb, ...), so the module is skipped when that stack is
absent. The trainer, data loading, and backbone are stubbed: these tests cover
config loading, the env overrides, and what gets handed to the HF trainer.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

run = pytest.importorskip("udonpred.training.run")

import yaml
from torch import nn
from transformers.trainer_utils import number_of_arguments

import udonpred.training.model.build_model as build_model

REPO_CONFIG = Path(__file__).resolve().parents[1] / "config"


def test_batch_size_fits_in_single_batch():
    # total <= max -> no gradient accumulation
    assert run.calculate_batch_sizes(4, 8) == (4, 1)


def test_batch_size_splits_into_accumulation_steps():
    # total > max -> (max_single, total // max_single)
    assert run.calculate_batch_sizes(16, 4) == (4, 4)


def test_batch_size_exact_multiple():
    assert run.calculate_batch_sizes(8, 8) == (8, 1)


# ---- run(): config loading, env overrides, dispatch ---------------------------

LOSS = {"class": "torch.nn.MSELoss", "output": "out_-1", "class_arguments": {}, "arguments": []}
CONFIG = {
    "config": {
        "cuda_devices": 0,
        "run_name": "base",
        "deepspeed_zero": False,
        "num_train_epochs": 3,
        "early_stopping": 2,
        "checkpoint": False,
        "logging": {"logging_steps": 5, "eval_strategy": "epoch"},
        "saving": {"save_strategy": "epoch", "save_total_limit": 1},
        "input_dim": 8,
        "output_dim": 1,
        "max_single_batch_size": 4,
        "heads": ["out"],
        "pre": {"out": []},
        "post": {"out": [["torch.nn.Sigmoid", {}]]},
        "outputs": ["out_-1"],
        "cluster": False,
        "filter": False,
        "min_length": 0,
        "hyperparameter_path": "config/architecture.yaml",
        "backbone": {
            "name": "Rostlab/ProstT5_fp16",
            "tokenizer_type": "T5Tokenizer",
            "model_type": "T5EncoderModel",
            "prefix_token": "<AA2fold>",
        },
        "embeddings": {"source": "auto"},
        "optim": {"backend": "optuna", "n_trials": 2, "direction": "minimize"},
    },
    "data": {
        "trizod": {"path": "data/split/trizod", "fraction": 1,
                   "losses": {"y": [LOSS]}, "metrics": {"y": [LOSS]}},
        "chezod": {"path": "data/split/chezod", "fraction": 0, "post": [],
                   "losses": {"y": [LOSS]}, "metrics": {"y": [LOSS]}},
    },
    "architecture": {
        "batch_size": 16,
        "learning_rate": 0.001,
        "lr_scheduler": "constant",
        "n_layers": 1,
        "layer_0": "torch.nn.Linear",
        "dim_0": 4,
        "activation_0": "torch.nn.LeakyReLU",
        "dropout_0": 0.0,
    },
    "optimize": {
        "learning_rate": {"type": "float", "values": [0.0001, 0.01]},
        "batch_size": {"type": "int", "values": [1, 128]},
        "lr_scheduler": {"type": "categorical", "values": ["cosine", "constant"]},
        "num_layers": {"type": "int", "values": [1, 3]},
        "layer_type": {"type": "categorical", "values": ["torch.nn.Linear"]},
        "layer_size": {"type": "int", "values": [4, 16]},
        "layer_params": {"torch.nn.Linear": {}},
        "activation_type": {"type": "categorical", "values": ["torch.nn.ReLU"]},
        "dropout_rate": {"type": "float", "values": [0.0, 0.5]},
        "lora": {"finetune": {"type": "bool", "values": [False]}},
    },
}


class FakeTrainer:
    """Records what run() hands the HF trainer, without training anything."""

    instances = []

    def __init__(self, config, **kwargs):
        self.config = config
        self.kwargs = kwargs
        self.calls = []
        FakeTrainer.instances.append(self)

    def remove_callback(self, callback):
        self.calls.append(("remove_callback", callback))

    def add_callback(self, callback):
        self.calls.append(("add_callback", type(callback).__name__))

    def train(self, resume_from_checkpoint=None):
        self.calls.append(("train", resume_from_checkpoint))

    def save_model(self, output_dir):
        self.calls.append(("save_model", output_dir))

    def hyperparameter_search(self, **kwargs):
        self.search_kwargs = kwargs
        return SimpleNamespace(run_id="7", hyperparameters={"learning_rate": 0.005})


class FakeTrial:
    """An optuna-like trial that always picks the lower bound / first choice."""

    def suggest_int(self, name, low, high):
        return low

    def suggest_float(self, name, low, high):
        return low

    def suggest_categorical(self, name, values):
        return values[0]


@pytest.fixture
def harness(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    for name, section in CONFIG.items():
        (config_dir / f"{name}.yaml").write_text(yaml.safe_dump(section))
    # cwd without a config/ dir, so hyperparameter_path must resolve via CONFIG_DIR
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setattr(run, "CONFIG_DIR", str(config_dir))
    for var in ("UDONPRED_TARGET", "UDONPRED_EMBEDDINGS_PLM", "UDONPRED_RUN_NAME"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("WANDB_PROJECT", raising=False)

    seen = {}
    monkeypatch.setattr(run, "get_data", lambda config: seen.setdefault(
        "data", {"train": "TRAIN", "validation": "VALID"}))
    monkeypatch.setattr(run, "DataCollator", lambda name, tokenizer: "COLLATOR")
    monkeypatch.setattr(run, "CustomTrainer", FakeTrainer)
    monkeypatch.setattr(run, "setup_wandb", lambda config, hp=None: None)

    def fake_load_model(config, finetune, lora_config, heads, output_keys):
        seen.setdefault("load_model", []).append(
            SimpleNamespace(config=config, finetune=finetune, lora=lora_config,
                            heads=heads, output_keys=output_keys))
        return "MODEL"

    monkeypatch.setattr(build_model, "load_model", fake_load_model)
    FakeTrainer.instances = []
    return seen


def _layer_types(module):
    return [type(layer).__name__ for layer in module.modules()][1:]


def test_repo_config_loads_every_section():
    config = {p.stem: yaml.safe_load(p.read_text()) for p in REPO_CONFIG.glob("*.yaml")}
    assert {"config", "data", "architecture", "optimize"} <= set(config)


def test_train_hands_the_trainer_the_configured_arguments(harness):
    run.run("train")

    (trainer,) = FakeTrainer.instances
    args = trainer.kwargs["args"]
    assert trainer.kwargs["model"] == "MODEL"
    assert trainer.kwargs["data_collator"] == "COLLATOR"
    assert trainer.kwargs["train_dataset"] == "TRAIN"
    assert trainer.kwargs["eval_dataset"] == "VALID"
    # batch_size 16 split over max_single_batch_size 4
    assert args.per_device_train_batch_size == 4
    assert args.gradient_accumulation_steps == 4
    assert args.learning_rate == 0.001
    assert args.num_train_epochs == 3
    assert args.output_dir == "checkpoints/base"
    assert args.load_best_model_at_end and args.metric_for_best_model == "eval_loss"
    assert [type(cb).__name__ for cb in trainer.kwargs["callbacks"]] == ["EarlyStoppingCallback"]
    assert ("train", False) in trainer.calls
    assert trainer.calls[-1] == ("save_model", "checkpoints/base/best")


def test_train_builds_the_heads_from_the_hyperparameters(harness):
    run.run("train")

    (call,) = harness["load_model"]
    assert call.finetune is False and call.lora == {}
    assert call.output_keys == {"out_-1"}
    assert _layer_types(call.heads["out"]) == [
        "Sequential", "Linear", "LeakyReLU", "Linear", "Sigmoid"]
    assert call.config["config"]["run_name"] == "base"


def test_env_selects_a_single_target_and_names_the_run(harness, monkeypatch):
    monkeypatch.setenv("UDONPRED_TARGET", "chezod")
    monkeypatch.setenv("UDONPRED_EMBEDDINGS_PLM", "frustraiseq")
    run.run("train")

    (trainer,) = FakeTrainer.instances
    config = trainer.config
    assert config["data"]["chezod"]["fraction"] == 1
    assert config["data"]["trizod"]["fraction"] == 0
    assert config["config"]["embeddings"]["plm"] == "frustraiseq"
    assert config["config"]["run_name"] == "chezod-frustraiseq"
    assert trainer.kwargs["args"].output_dir == "checkpoints/chezod-frustraiseq"


def test_a_single_target_brings_its_own_output_activation(harness, monkeypatch):
    # chezod is an unbounded regression: its data entry drops the config-wide Sigmoid
    monkeypatch.setenv("UDONPRED_TARGET", "chezod")
    run.run("train")
    (call,) = harness["load_model"]
    assert _layer_types(call.heads["out"]) == ["Sequential", "Linear", "LeakyReLU", "Linear"]


def test_explicit_run_name_wins(harness, monkeypatch):
    monkeypatch.setenv("UDONPRED_TARGET", "chezod")
    monkeypatch.setenv("UDONPRED_EMBEDDINGS_PLM", "frustraiseq")
    monkeypatch.setenv("UDONPRED_RUN_NAME", "custom")
    run.run("train")
    assert FakeTrainer.instances[0].config["config"]["run_name"] == "custom"


def _tag_config(tags):
    path = Path(run.CONFIG_DIR) / "config.yaml"
    path.write_text(yaml.safe_dump({**CONFIG["config"], "experiment_tags": tags}))


def test_experiment_tags_extend_a_derived_run_name(harness, monkeypatch):
    _tag_config(["exp1", "exp2"])
    monkeypatch.setenv("UDONPRED_TARGET", "chezod")
    monkeypatch.setenv("UDONPRED_EMBEDDINGS_PLM", "prostt5")
    assert run.load_config()["config"]["run_name"] == "chezod-prostt5-exp1-exp2"


def test_experiment_tags_leave_an_explicit_run_name_alone(harness, monkeypatch):
    _tag_config(["exp1"])
    monkeypatch.setenv("UDONPRED_RUN_NAME", "custom")
    assert run.load_config()["config"]["run_name"] == "custom"


def _wandb_init_kwargs(monkeypatch, config):
    seen = {}
    monkeypatch.setattr(run.wandb, "init", lambda **kwargs: seen.update(kwargs))
    run.setup_wandb({**config, "config": {**config["config"], "wandb": {"project": "P", "log_model": "end"}}})
    return seen


def test_wandb_run_carries_the_experiment_tags(monkeypatch):
    config = {"config": {"run_name": "chezod-prostt5-exp1", "experiment_tags": ["exp1"]}}
    kwargs = _wandb_init_kwargs(monkeypatch, config)
    assert kwargs["name"] == "chezod-prostt5-exp1"
    assert kwargs["tags"] == ["exp1"]


def test_untagged_wandb_run_is_the_baseline(monkeypatch):
    kwargs = _wandb_init_kwargs(monkeypatch, {"config": {"run_name": "chezod-prostt5"}})
    assert kwargs["tags"] == ["baseline"]


def test_unknown_target_is_rejected(harness, monkeypatch):
    monkeypatch.setenv("UDONPRED_TARGET", "nope")
    with pytest.raises(ValueError, match="UDONPRED_TARGET='nope'"):
        run.run("train")


def test_hyperparameter_path_falls_back_to_the_config_dir(harness):
    run.run("train")
    hp_path = FakeTrainer.instances[0].config["config"]["hyperparameter_path"]
    assert hp_path == str(Path(run.CONFIG_DIR) / "architecture.yaml")


def test_run_leaves_no_global_config(harness):
    run.run("train")
    assert "config" not in sys.modules


def test_load_config_reads_an_explicit_dir(harness, tmp_path):
    config = run.load_config(str(tmp_path / "config"))
    assert set(config) == {"config", "data", "architecture", "optimize"}
    assert config["architecture"]["dim_0"] == 4


def test_invalid_mode_is_rejected(harness):
    with pytest.raises(ValueError, match="Invalid mode"):
        run.run("evaluate")


def test_optimize_hands_hf_callables_it_can_call(harness, tmp_path):
    run.run("optimize")

    (trainer,) = FakeTrainer.instances
    model_init = trainer.kwargs["model_init"]
    hp_space = trainer.search_kwargs["hp_space"]
    # HF calls model_init(trial) and hp_space(trial) with a single argument
    assert number_of_arguments(model_init) == 1
    assert number_of_arguments(hp_space) == 1

    # HF's first model_init call has no trial: a bare linear head per output
    assert model_init(None) == "MODEL"
    assert _layer_types(harness["load_model"][-1].heads["out"]) == ["Linear", "Sigmoid"]

    trial = FakeTrial()
    assert hp_space(trial) == {
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.0001,
        "lr_scheduler_type": "cosine",
    }
    model_init(trial)
    assert _layer_types(harness["load_model"][-1].heads["out"]) == [
        "Sequential", "Linear", "ReLU", "Linear", "Sigmoid"]

    best = Path("optimized_parameters/base/7.yaml")
    assert yaml.safe_load(best.read_text()) == {"learning_rate": 0.005}
