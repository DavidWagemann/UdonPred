import pytest

pytest.importorskip("huggingface_hub")

from udonpred.utils import publish


EXCLUDED = "trizod2"


def _make_target(root, name):
    d = root / name
    d.mkdir(parents=True)
    for split in ("train", "valid", "test"):
        (d / f"{split}.jsonl").write_text('{"id": "1", "x_0": "MK"}\n')
        (d / f"{split}.fasta").write_text(">1\nMK\n")
    return d


def test_dataset_card_lists_configs():
    card = publish.dataset_card(["trizod", "chezod"])
    assert "config_name: trizod" in card
    assert "config_name: chezod" in card
    assert "split: validation" in card and "valid.jsonl" in card


def test_publish_dataset_uploads_files_and_card(tmp_path, monkeypatch):
    data = tmp_path / "split"
    _make_target(data, "trizod")
    _make_target(data, EXCLUDED)  # must be skipped

    uploaded = []

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            uploaded.append(("create_repo", repo_id, kw.get("repo_type")))

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, **kw):
            uploaded.append(("upload_file", path_in_repo, repo_type))

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    publish.publish_dataset(str(data), repo="udonpred/datasets")

    paths = {p for kind, p, *_ in uploaded if kind == "upload_file"}
    assert "trizod/train.jsonl" in paths
    assert "trizod/valid.fasta" in paths
    assert "README.md" in paths
    assert not any(p.startswith(f"{EXCLUDED}/") for p in paths)  # excluded


def test_publish_embeddings_targets_expected_path(tmp_path, monkeypatch):
    h5 = tmp_path / "test.h5"
    h5.write_bytes(b"x")
    seen = {}

    class FakeApi:
        def create_repo(self, repo_id, **kw):
            pass

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, **kw):
            seen["path_in_repo"] = path_in_repo
            seen["repo_type"] = repo_type

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)

    publish.publish_embeddings(h5, target="trizod", split="test", plm="prostt5")
    assert seen["path_in_repo"] == "trizod/embeddings/prostt5/test.h5"
    assert seen["repo_type"] == "dataset"
