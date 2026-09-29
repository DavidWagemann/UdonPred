from pathlib import Path

import pytest

import udonpred.datasets as ds
from udonpred.datasets import (
    DEFAULT_DATASET_REPO,
    DEFAULT_DATASET_REVISION,
    parse_embeddings_ref,
    plm_slug,
    resolve_embeddings,
    resolve_embeddings_ref,
    resolve_split_file,
)


@pytest.fixture
def capture_download(monkeypatch):
    calls = []

    def fake(repo_id, filename, revision):
        calls.append((repo_id, filename, revision))
        return Path("/cache") / repo_id.replace("/", "__") / filename

    monkeypatch.setattr(ds, "_hf_download", fake)
    return calls


def test_plm_slug_alias():
    assert plm_slug("Rostlab/ProstT5_fp16") == "prostt5"


def test_plm_slug_sanitizes_other_models():
    assert plm_slug("facebook/esm2_t33_650M_UR50D") == "esm2-t33-650m-ur50d"


def test_parse_ref_local_path_is_none(tmp_path):
    f = tmp_path / "emb.h5"
    f.write_bytes(b"")
    assert parse_embeddings_ref(str(f)) is None


def test_parse_ref_without_colon_is_none():
    assert parse_embeddings_ref("some/local/path.h5") is None


def test_parse_ref_hub_reference():
    assert parse_embeddings_ref("udonpred/datasets:trizod/embeddings/prostt5/test.h5") == (
        "udonpred/datasets",
        "trizod/embeddings/prostt5/test.h5",
        None,
    )


def test_parse_ref_with_revision():
    assert parse_embeddings_ref("org/name:a/b.h5@v2") == ("org/name", "a/b.h5", "v2")


def test_resolve_ref_local_passthrough(tmp_path, capture_download):
    f = tmp_path / "emb.h5"
    f.write_bytes(b"")
    assert resolve_embeddings_ref(str(f)) == f
    assert capture_download == []


def test_resolve_ref_downloads_hub(capture_download):
    resolve_embeddings_ref("udonpred/datasets:trizod/embeddings/prostt5/test.h5")
    assert capture_download == [
        ("udonpred/datasets", "trizod/embeddings/prostt5/test.h5", DEFAULT_DATASET_REVISION)
    ]


def test_resolve_embeddings_builds_path(capture_download):
    resolve_embeddings("trizod", "valid", "prostt5")
    assert capture_download == [
        (DEFAULT_DATASET_REPO, "trizod/embeddings/prostt5/valid.h5", DEFAULT_DATASET_REVISION)
    ]


def test_resolve_split_file_builds_path(capture_download):
    resolve_split_file("chezod", "train", "fasta")
    assert capture_download == [
        (DEFAULT_DATASET_REPO, "chezod/train.fasta", DEFAULT_DATASET_REVISION)
    ]


@pytest.fixture
def dataset_env(monkeypatch):
    monkeypatch.setenv("UDONPRED_DATASET_REPO", "myorg/data")
    monkeypatch.setenv("UDONPRED_DATASET_REVISION", "v1")


def test_resolve_embeddings_honors_env_overrides(dataset_env, capture_download):
    resolve_embeddings("trizod", "test", "prostt5")
    assert capture_download == [("myorg/data", "trizod/embeddings/prostt5/test.h5", "v1")]


def test_resolve_split_file_honors_env_overrides(dataset_env, capture_download):
    resolve_split_file("trizod", "valid")
    assert capture_download == [("myorg/data", "trizod/valid.jsonl", "v1")]


def test_explicit_repo_and_revision_beat_env(dataset_env, capture_download):
    resolve_embeddings("trizod", "test", "prostt5", repo="other/data", revision="main")
    resolve_split_file("trizod", "test", repo="other/data", revision="main")
    assert capture_download == [
        ("other/data", "trizod/embeddings/prostt5/test.h5", "main"),
        ("other/data", "trizod/test.jsonl", "main"),
    ]


def test_ref_revision_is_not_affected_by_dataset_env(dataset_env, capture_download):
    # a Hub reference names its own repo; only its revision falls back to the default
    resolve_embeddings_ref("org/name:a/b.h5")
    assert capture_download == [("org/name", "a/b.h5", DEFAULT_DATASET_REVISION)]
