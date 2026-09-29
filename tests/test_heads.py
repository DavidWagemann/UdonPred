"""Unit tests for :mod:`udonpred.heads` (local-or-Hub head resolution).

The Hub download branch is monkeypatched, so these run offline and never import
huggingface_hub.
"""

from pathlib import Path

import pytest

import udonpred.heads as heads
from udonpred.heads import (
    DEFAULT_HEADS_REPO,
    DEFAULT_HEADS_REVISION,
    discover_targets,
    resolve_model_dir,
    resolve_targets,
)


@pytest.fixture
def clean_env(monkeypatch):
    """Ensure none of the UDONPRED_HEADS_* overrides leak in from the environment."""
    for var in ("UDONPRED_HEADS_DIR", "UDONPRED_HEADS_REPO", "UDONPRED_HEADS_REVISION"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def capture_download(monkeypatch):
    """Replace the Hub download with a recorder that returns a sentinel path."""
    calls = []

    def fake_download(repo_id, revision):
        calls.append((repo_id, revision))
        return Path("/cache") / repo_id.replace("/", "__") / revision

    monkeypatch.setattr(heads, "_download_heads", fake_download)
    return calls


# ---- local resolution (no download) ------------------------------------------

def test_existing_local_dir_used_as_is(tmp_path, clean_env, capture_download):
    (tmp_path / "trizod.onnx").write_bytes(b"")
    assert resolve_model_dir(str(tmp_path)) == tmp_path
    assert capture_download == []  # never touched the Hub


def test_env_dir_used_when_source_none(tmp_path, monkeypatch, capture_download):
    monkeypatch.setenv("UDONPRED_HEADS_DIR", str(tmp_path))
    assert resolve_model_dir(None) == tmp_path
    assert capture_download == []


# ---- Hub resolution (download branch) ----------------------------------------

def test_none_source_falls_back_to_pinned_default(clean_env, capture_download):
    resolve_model_dir(None)
    assert capture_download == [(DEFAULT_HEADS_REPO, DEFAULT_HEADS_REVISION)]


def test_repo_id_source_is_downloaded(clean_env, capture_download):
    resolve_model_dir("udonpred/prediction-heads", "v9.9.9")
    assert capture_download == [("udonpred/prediction-heads", "v9.9.9")]


def test_nonexistent_local_path_treated_as_repo_id(clean_env, capture_download):
    resolve_model_dir("not/a/real/dir")
    assert capture_download == [("not/a/real/dir", DEFAULT_HEADS_REVISION)]


def test_env_overrides_repo_and_revision(monkeypatch, capture_download):
    monkeypatch.delenv("UDONPRED_HEADS_DIR", raising=False)
    monkeypatch.setenv("UDONPRED_HEADS_REPO", "myorg/heads")
    monkeypatch.setenv("UDONPRED_HEADS_REVISION", "dev")
    resolve_model_dir(None)
    assert capture_download == [("myorg/heads", "dev")]


def test_explicit_revision_beats_default(clean_env, capture_download):
    resolve_model_dir(None, revision="main")
    assert capture_download == [(DEFAULT_HEADS_REPO, "main")]


# ---- target discovery / validation -------------------------------------------

@pytest.fixture
def heads_dir(tmp_path):
    for name in ("trizod", "chezod", "atlas"):
        (tmp_path / f"{name}.onnx").write_bytes(b"")
    (tmp_path / "README.md").write_text("not a head")
    return tmp_path


def test_discover_targets_lists_onnx_stems_sorted(heads_dir):
    assert discover_targets(heads_dir) == ["atlas", "chezod", "trizod"]


def test_resolve_targets_expands_all(heads_dir):
    assert resolve_targets(heads_dir, ["all"]) == ["atlas", "chezod", "trizod"]


def test_resolve_targets_keeps_requested_order(heads_dir):
    assert resolve_targets(heads_dir, ["trizod", "atlas"]) == ["trizod", "atlas"]


def test_resolve_targets_rejects_missing_head(heads_dir):
    with pytest.raises(ValueError, match="ONNX head not found"):
        resolve_targets(heads_dir, ["trizod", "plddt"])


def test_resolve_all_in_empty_dir_raises(tmp_path):
    with pytest.raises(ValueError, match="No .onnx heads found"):
        resolve_targets(tmp_path, ["all"])
