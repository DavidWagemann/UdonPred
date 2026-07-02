# Hub datasets & per-pLM embeddings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute UdonPred's per-target datasets (train/valid/test FASTA+jsonl) and precomputed per-pLM embeddings through a single public `udonpred/datasets` Hub repo, consumed by training (precomputed-first) and inference (Hub-reference `--embeddings`).

**Architecture:** A lean, torch-free `udonpred/datasets.py` resolves dataset/embedding files from a local path or the Hub (lazy `huggingface_hub`), mirroring `udonpred/heads.py`. Upload tooling lives in `udonpred/utils/publish.py`. Training's `build_datasets` gains a precomputed-first path; inference's `--embeddings` gains Hub-reference support.

**Tech Stack:** Python 3.13, `huggingface_hub` (hub extra), `datasets` + `h5py` + `numpy` (training/consume), `pytest`, hatchling, uv.

## Global Constraints

- Repo: `udonpred/datasets`, **public**, HF **dataset** repo, revision default `main` (overridable; no tags).
- Targets: `trizod, chezod, disprot, pdbflex, plddt, softdis, atlas`. **`trizod2` excluded.**
- Embeddings keyed by the jsonl **`id`** (not the underscore-stripped FASTA header); stored at `<target>/embeddings/<plm-slug>/<split>.h5`; splits named `train`/`valid`/`test` for files.
- `udonpred/datasets.py` must be **torch-free and import-free of `huggingface_hub`** (lazy only), so the CAID inference path stays lean/offline; the boundary test enforces this.
- No embedding generation in this plan (needs GPU + `[embedding]`); only upload/consume plumbing, tested with mock/small data.
- Config additions kept minimal (KISS), optional, backwards-compatible.
- Every commit message ends with the `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>` trailer.
- Run tests with `.venv/bin/python -m pytest` (do not `uv run`, to avoid pruning the dev venv).

---

### Task 1: Lean consume module `udonpred/datasets.py`

**Files:**
- Create: `src/udonpred/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Produces:
  - `DEFAULT_DATASET_REPO = "udonpred/datasets"`, `DEFAULT_DATASET_REVISION = "main"`
  - `plm_slug(name: str) -> str`
  - `parse_embeddings_ref(source: str) -> tuple[str, str, str | None] | None`
  - `resolve_embeddings_ref(source: str, revision: str | None = None) -> Path`
  - `resolve_embeddings(target: str, split: str, plm: str, repo: str | None = None, revision: str | None = None) -> Path`
  - `resolve_split_file(target: str, split: str, kind: str = "jsonl", repo: str | None = None, revision: str | None = None) -> Path`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_datasets.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'udonpred.datasets'`)

- [ ] **Step 3: Write the module**

```python
# src/udonpred/datasets.py
"""Resolve UdonPred datasets and per-pLM embeddings from a local path or the Hub.

Datasets live in the public Hugging Face dataset repo ``udonpred/datasets``
(one config per target; embeddings under ``<target>/embeddings/<plm>/``). This
module is the lean consume side: pure path logic plus a lazily-imported
``huggingface_hub`` download, so the CAID inference path stays torch- and
hub-free until a Hub file is actually fetched. Uploading lives in
:mod:`udonpred.utils.publish`.
"""

import os
import re
from pathlib import Path

DEFAULT_DATASET_REPO = "udonpred/datasets"
DEFAULT_DATASET_REVISION = "main"

ENV_DATASET_REPO = "UDONPRED_DATASET_REPO"
ENV_DATASET_REVISION = "UDONPRED_DATASET_REVISION"

# Curated pLM name -> slug aliases; everything else is sanitized from the id.
_PLM_ALIASES = {
    "rostlab/prostt5_fp16": "prostt5",
    "rostlab/prostt5": "prostt5",
}


def plm_slug(name: str) -> str:
    """Filesystem-safe slug for a protein language model id."""
    key = name.lower()
    if key in _PLM_ALIASES:
        return _PLM_ALIASES[key]
    tail = name.split("/")[-1].lower()
    return re.sub(r"[^a-z0-9]+", "-", tail).strip("-")


def parse_embeddings_ref(source: str) -> "tuple[str, str, str | None] | None":
    """Parse a ``repo_id:path_in_repo[@revision]`` Hub reference.

    Returns ``None`` when ``source`` is a local path (an existing file, or a
    string that does not look like ``org/name:path``).
    """
    if os.path.exists(source):
        return None
    if ":" not in source:
        return None
    repo_id, _, rest = source.partition(":")
    if "/" not in repo_id:
        return None
    path_in_repo, _, revision = rest.partition("@")
    return repo_id, path_in_repo, (revision or None)


def _hf_download(repo_id: str, filename: str, revision: str) -> Path:
    """Download one file from a Hub *dataset* repo, returning the cached path."""
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise ModuleNotFoundError(
            f"Fetching {filename!r} from the Hub ({repo_id!r}) requires "
            "huggingface_hub. Install it with `pip install 'udonpred-comp[hub]'`, "
            "or pass a local file instead."
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repo_id, filename=filename, revision=revision, repo_type="dataset"
        )
    )


def resolve_embeddings_ref(source: str, revision: str | None = None) -> Path:
    """Return a local ``.h5``/``.npy`` path from a local file or a Hub reference."""
    parsed = parse_embeddings_ref(source)
    if parsed is None:
        return Path(source)
    repo_id, path_in_repo, ref_revision = parsed
    rev = revision or ref_revision or DEFAULT_DATASET_REVISION
    return _hf_download(repo_id, path_in_repo, rev)


def resolve_embeddings(
    target: str,
    split: str,
    plm: str,
    repo: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download ``<target>/embeddings/<plm>/<split>.h5`` from the dataset repo."""
    repo = repo or os.environ.get(ENV_DATASET_REPO) or DEFAULT_DATASET_REPO
    rev = revision or os.environ.get(ENV_DATASET_REVISION) or DEFAULT_DATASET_REVISION
    return _hf_download(repo, f"{target}/embeddings/{plm}/{split}.h5", rev)


def resolve_split_file(
    target: str,
    split: str,
    kind: str = "jsonl",
    repo: str | None = None,
    revision: str | None = None,
) -> Path:
    """Download ``<target>/<split>.<kind>`` (``jsonl``/``fasta``) from the repo."""
    repo = repo or os.environ.get(ENV_DATASET_REPO) or DEFAULT_DATASET_REPO
    rev = revision or os.environ.get(ENV_DATASET_REVISION) or DEFAULT_DATASET_REVISION
    return _hf_download(repo, f"{target}/{split}.{kind}", rev)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_datasets.py -q`
Expected: PASS (10 passed)

- [ ] **Step 5: Commit**

```bash
git add src/udonpred/datasets.py tests/test_datasets.py
git commit -m "feat: resolve datasets and per-pLM embeddings from the Hub

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Wire Hub-reference `--embeddings` into inference

**Files:**
- Modify: `src/udonpred/caid/predict.py` (imports; `run()` resolves `embeddings`)
- Modify: `src/udonpred/embedding/predict.py` (n/a — it computes embeddings; skip)
- Modify: `tests/test_import_boundary.py` (add `udonpred.datasets` to the guard)
- Test: `tests/test_caid_hub_embeddings.py`

**Interfaces:**
- Consumes: `udonpred.datasets.resolve_embeddings_ref` (Task 1).

Note: only `udonpred-caid` consumes precomputed embeddings via `--embeddings`;
`udonpred-predict` computes them on the fly, so it needs no change here.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_caid_hub_embeddings.py
import numpy as np

import udonpred.datasets as ds
from udonpred.caid import predict as caid


def test_run_resolves_hub_embeddings_reference(tmp_path, monkeypatch, weights_dir):
    # A fake .h5 the "Hub reference" resolves to.
    import h5py

    emb_path = tmp_path / "test.h5"
    with h5py.File(emb_path, "w") as f:
        f["seq1"] = np.random.randn(10, 1024).astype(np.float32)
    (tmp_path / "in.fasta").write_text(">seq1\nMKTAYIAKQR\n")

    seen = {}

    def fake_resolve(source, revision=None):
        seen["source"] = source
        return emb_path

    monkeypatch.setattr(caid, "resolve_embeddings_ref", fake_resolve)

    caid.run(
        fasta=str(tmp_path / "in.fasta"),
        model_dir=str(weights_dir),
        embeddings="udonpred/datasets:trizod/embeddings/prostt5/test.h5",
        target=["trizod"],
        output_path=str(tmp_path / "out"),
        device="cpu",
        threads=None,
        smooth=1.5,
    )

    assert seen["source"] == "udonpred/datasets:trizod/embeddings/prostt5/test.h5"
    assert (tmp_path / "out" / "udonpred_trizod.caid").read_text().startswith(">seq1\n")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_caid_hub_embeddings.py -q`
Expected: FAIL (`AttributeError: module 'udonpred.caid.predict' has no attribute 'resolve_embeddings_ref'`)

- [ ] **Step 3: Wire the resolver into `caid/predict.py`**

Add to the imports block (near the other `udonpred` imports):

```python
from udonpred.datasets import resolve_embeddings_ref
```

In `run(...)`, replace:

```python
    source = load_precomputed_embeddings(embeddings)
```

with:

```python
    # `embeddings` may be a local .h5/.npy path or a Hub reference
    # (repo_id:path_in_repo[@revision]).
    source = load_precomputed_embeddings(str(resolve_embeddings_ref(embeddings)))
```

Update the `--embeddings` help text in `build_parser()` to mention Hub refs:

```python
        help="Path to precomputed ProstT5 embeddings (.npy single sequence, "
        "or .h5 keyed by FASTA header), or a Hub reference like "
        "udonpred/datasets:trizod/embeddings/prostt5/test.h5.",
```

- [ ] **Step 4: Extend the boundary guard**

In `tests/test_import_boundary.py`, add `import udonpred.datasets` to the
subprocess `code` string (after `import udonpred.heads`). The existing torch /
huggingface_hub assertions then also cover `udonpred.datasets`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_caid_hub_embeddings.py tests/test_import_boundary.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/udonpred/caid/predict.py tests/test_caid_hub_embeddings.py tests/test_import_boundary.py
git commit -m "feat: accept Hub references for --embeddings in the CAID runner

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Upload tooling `udonpred/utils/publish.py`

**Files:**
- Create: `src/udonpred/utils/publish.py`
- Modify: `pyproject.toml` (add `udonpred-publish-dataset` script)
- Test: `tests/test_publish.py`

**Interfaces:**
- Consumes: `udonpred.datasets.plm_slug`, `DEFAULT_DATASET_REPO` (Task 1).
- Produces:
  - `dataset_card(targets: list[str]) -> str`
  - `publish_dataset(data_dir, repo=None, targets=None) -> None`
  - `publish_embeddings(h5_path, target, split, plm, repo=None) -> None`
  - `main()` (console entry `udonpred-publish-dataset`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_publish.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_publish.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'udonpred.utils.publish'`)

- [ ] **Step 3: Write the module**

```python
# src/udonpred/utils/publish.py
"""Upload UdonPred datasets and per-pLM embeddings to the Hub.

Publishes the per-target train/valid/test FASTA+jsonl into a single dataset
repo (with a per-target ``configs:`` card) and uploads precomputed embeddings
under ``<target>/embeddings/<plm>/<split>.h5``. Uses the ambient
``huggingface-cli login``; ``huggingface_hub`` is imported lazily.
"""

import argparse
from pathlib import Path

from udonpred.datasets import DEFAULT_DATASET_REPO

# trizod2 has a different on-disk layout and is handled separately.
EXCLUDED_TARGETS = {"trizod2"}
SPLITS = ("train", "valid", "test")
# File split name -> HF split name for the dataset card.
_CARD_SPLIT = {"train": "train", "valid": "validation", "test": "test"}


def dataset_card(targets: list[str]) -> str:
    """Build a dataset-card README with one ``configs:`` entry per target."""
    lines = ["---", "configs:"]
    for target in targets:
        lines.append(f"  - config_name: {target}")
        lines.append("    data_files:")
        for split in SPLITS:
            lines.append(f"      - split: {_CARD_SPLIT[split]}")
            lines.append(f"        path: {target}/{split}.jsonl")
    lines += [
        "---",
        "",
        "# UdonPred datasets",
        "",
        "Per-target protein intrinsic-disorder datasets for "
        "[UdonPred](https://github.com/davidwagemann/udonpred): `train`/`valid`/"
        "`test` as jsonl (`{id, y, x_0}`) and FASTA, plus precomputed per-pLM "
        "embeddings under `<target>/embeddings/<plm>/<split>.h5` (keyed by jsonl id).",
        "",
    ]
    return "\n".join(lines)


def _discover_targets(data_dir: Path) -> list[str]:
    return sorted(
        d.name
        for d in data_dir.iterdir()
        if d.is_dir()
        and d.name not in EXCLUDED_TARGETS
        and not d.name.startswith(".")
        and (d / "train.jsonl").exists()
    )


def publish_dataset(data_dir, repo: str | None = None, targets=None) -> None:
    """Upload each target's train/valid/test jsonl+fasta and a configs card."""
    from huggingface_hub import HfApi

    data_dir = Path(data_dir)
    repo = repo or DEFAULT_DATASET_REPO
    targets = targets or _discover_targets(data_dir)
    if not targets:
        raise ValueError(f"No datasets found under {data_dir}")

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)

    for target in targets:
        for split in SPLITS:
            for kind in ("jsonl", "fasta"):
                src = data_dir / target / f"{split}.{kind}"
                if src.exists():
                    api.upload_file(
                        path_or_fileobj=str(src),
                        path_in_repo=f"{target}/{split}.{kind}",
                        repo_id=repo,
                        repo_type="dataset",
                    )
        print(f"Uploaded {target}")

    api.upload_file(
        path_or_fileobj=dataset_card(list(targets)).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"Published dataset -> https://huggingface.co/datasets/{repo}")


def publish_embeddings(
    h5_path, target: str, split: str, plm: str, repo: str | None = None
) -> None:
    """Upload one embeddings ``.h5`` to ``<target>/embeddings/<plm>/<split>.h5``."""
    from huggingface_hub import HfApi

    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise ValueError(f"Embeddings file not found: {h5_path}")
    repo = repo or DEFAULT_DATASET_REPO

    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(h5_path),
        path_in_repo=f"{target}/embeddings/{plm}/{split}.h5",
        repo_id=repo,
        repo_type="dataset",
    )
    print(f"Uploaded {target}/{split} embeddings ({plm}) -> {repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish UdonPred datasets (train/valid/test jsonl+fasta) to the Hub."
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/split",
        help="Directory with per-target subfolders (default: data/split).",
    )
    parser.add_argument("--repo", default=DEFAULT_DATASET_REPO)
    parser.add_argument(
        "--targets", nargs="+", default=None, help="Targets to upload (default: all)."
    )
    args = parser.parse_args()
    publish_dataset(args.data_dir, repo=args.repo, targets=args.targets)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Register the console script in `pyproject.toml`**

Under `[project.scripts]`, add:

```toml
udonpred-publish-dataset = "udonpred.utils.publish:main"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_publish.py -q`
Expected: PASS (3 passed)

- [ ] **Step 6: Commit**

```bash
git add src/udonpred/utils/publish.py tests/test_publish.py pyproject.toml
git commit -m "feat: add udonpred-publish-dataset and embedding upload helpers

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `udonpred-embed --push-to-hub`

**Files:**
- Modify: `src/udonpred/utils/embed.py` (add `--push-to-hub`, `--dataset`, `--split`, `--repo`; call `publish_embeddings`)
- Test: `tests/test_embed.py` (add a monkeypatched push test)

**Interfaces:**
- Consumes: `udonpred.utils.publish.publish_embeddings`, `udonpred.datasets.plm_slug`, `udonpred.embedding.backbone.BACKBONE_NAME`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_embed.py`:

```python
def test_embed_push_to_hub_calls_publish(tmp_path, monkeypatch):
    # generate() needs torch; instead test the push wiring in isolation.
    from udonpred.utils import embed

    called = {}

    def fake_publish(h5_path, target, split, plm, repo=None):
        called.update(target=target, split=split, plm=plm, repo=repo)

    monkeypatch.setattr(embed, "publish_embeddings", fake_publish)

    h5 = tmp_path / "e.h5"
    h5.write_bytes(b"x")
    embed.push_embeddings_to_hub(
        str(h5), dataset="trizod", split="test", backbone_name="Rostlab/ProstT5_fp16"
    )
    assert called == {"target": "trizod", "split": "test", "plm": "prostt5", "repo": None}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_embed.py::test_embed_push_to_hub_calls_publish -q`
Expected: FAIL (`AttributeError: module 'udonpred.utils.embed' has no attribute 'push_embeddings_to_hub'`)

- [ ] **Step 3: Add the push wiring to `embed.py`**

Add imports near the top:

```python
from udonpred.datasets import plm_slug
from udonpred.utils.publish import publish_embeddings
```

Add a small helper (module level):

```python
def push_embeddings_to_hub(
    output: str, dataset: str, split: str, backbone_name: str, repo: str | None = None
) -> None:
    """Upload a generated embeddings .h5 to the dataset repo for a pLM."""
    publish_embeddings(
        output, target=dataset, split=split, plm=plm_slug(backbone_name), repo=repo
    )
```

In `main()`, add arguments:

```python
    parser.add_argument("--push-to-hub", action="store_true",
        help="Upload the generated .h5 to the udonpred/datasets repo.")
    parser.add_argument("--dataset", default=None,
        help="Target name for --push-to-hub (e.g. trizod).")
    parser.add_argument("--split", default=None, choices=["train", "valid", "test"],
        help="Split name for --push-to-hub.")
    parser.add_argument("--repo", default=None,
        help="Override the dataset repo for --push-to-hub.")
```

After the `generate(...)` call in `main()`, add:

```python
    if args.push_to_hub:
        if not (args.dataset and args.split):
            parser.error("--push-to-hub requires --dataset and --split")
        push_embeddings_to_hub(
            args.output, args.dataset, args.split, BACKBONE_NAME, args.repo
        )
```

(`BACKBONE_NAME` is already imported in `embed.py`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_embed.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/udonpred/utils/embed.py tests/test_embed.py
git commit -m "feat: udonpred-embed --push-to-hub uploads embeddings to the dataset repo

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Precomputed-embedding attach helper

**Files:**
- Create: `src/udonpred/training/model/embeddings.py`
- Test: `tests/test_precomputed_embeddings.py`

**Interfaces:**
- Produces: `attach_precomputed_embeddings(dataset_split, h5_path, id_column="id", column="embedding_0") -> Dataset`

This is the torch-free core of the precomputed-first training path (uses
`datasets` + `h5py` + `numpy`, all available), unit-testable on its own.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_precomputed_embeddings.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_precomputed_embeddings.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'udonpred.training.model.embeddings'`)

- [ ] **Step 3: Write the helper**

```python
# src/udonpred/training/model/embeddings.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_precomputed_embeddings.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/udonpred/training/model/embeddings.py tests/test_precomputed_embeddings.py
git commit -m "feat: add precomputed-embedding attach helper for training

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: Precomputed-first wiring in `build_datasets`

**Files:**
- Modify: `src/udonpred/training/model/data.py` (`build_datasets`)
- Modify: `src/udonpred/training/config/config.yaml` (minimal `embeddings` block)
- Modify: `src/udonpred/training/config/data.yaml` (optional global `datasets` repo)

**Interfaces:**
- Consumes: `udonpred.datasets.{resolve_embeddings, plm_slug}`,
  `udonpred.training.model.embeddings.attach_precomputed_embeddings` (Tasks 1, 5).

This task changes training code that requires the full `[training]` stack
(torch/transformers/datasets), which is **not installed here**, so it is
covered by the isolated Task 5 helper test plus a manual review, not a new
executable test. Keep the change minimal and behind config.

- [ ] **Step 1: Add the precomputed-first branch to `build_datasets`**

`build_datasets(path, backbone_model, ..., dataset_name=None, plm=None,
embeddings_source="auto", cache_dir=...)`. After `ds = load_dataset("json", ...)`
and setting `dataset_name`, insert, before the tokenize/embed `ds.map`:

```python
    from udonpred.datasets import resolve_embeddings
    from udonpred.training.model.embeddings import attach_precomputed_embeddings

    used_precomputed = False
    if embeddings_source in ("auto", "hub") and plm is not None:
        try:
            file_split = {"train": "train", "validation": "valid", "test": "test"}
            attached = {}
            for split in ds.keys():
                h5 = resolve_embeddings(dataset_name, file_split[split], plm)
                attached[split] = attach_precomputed_embeddings(ds[split], str(h5))
            from datasets import DatasetDict
            ds = DatasetDict(attached)
            ds = ds.with_format("torch")
            used_precomputed = True
        except Exception as exc:
            if embeddings_source == "hub":
                raise
            print(f"Precomputed embeddings unavailable ({exc}); computing on the fly.")

    if not used_precomputed:
        # ... existing SQLite tokenize+embed path unchanged ...
```

Wrap the existing cache/`process_fn`/`ds.map`/`with_format` block in the
`if not used_precomputed:` branch. `plm` is passed from the caller as
`plm_slug(backbone.name)`.

- [ ] **Step 2: Thread `plm`/`embeddings_source` from the caller**

Where `build_datasets` is called (in `get_datasets`), pass
`dataset_name=os.path.basename(path)`, `plm=plm_slug(config["config"]["backbone"]["name"])`,
and `embeddings_source=config["config"].get("embeddings", {}).get("source", "auto")`.
Add `from udonpred.datasets import plm_slug` at the top of `data.py`.

- [ ] **Step 3: Add the minimal config block**

In `config/config.yaml`, add under the top-level `config:` mapping:

```yaml
  # Embeddings: "auto" prefers precomputed from the Hub (per backbone pLM),
  # falling back to on-the-fly compute; "compute" always computes locally.
  embeddings:
    source: auto
```

In `config/data.yaml`, document (comment only) that per-dataset `path` basenames
are used as the Hub config/target names.

- [ ] **Step 4: Sanity-check import wiring (no torch needed)**

Run: `.venv/bin/python -c "import ast; ast.parse(open('src/udonpred/training/model/data.py').read()); print('data.py parses')"`
Expected: `data.py parses`

- [ ] **Step 5: Commit**

```bash
git add src/udonpred/training/model/data.py src/udonpred/training/config/config.yaml src/udonpred/training/config/data.yaml
git commit -m "feat: prefer precomputed Hub embeddings in build_datasets

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 7: One-time dataset upload + docs

**Files:**
- Modify: `README.md` (dataset consumption, `--embeddings` Hub refs, publish flow)

- [ ] **Step 1: Publish the datasets to the Hub (operational, run once)**

Run: `.venv/bin/python -m udonpred.utils.publish data/split --repo udonpred/datasets`
Expected: uploads the 7 targets (trizod2 skipped) + `README.md`; prints the repo URL.

- [ ] **Step 2: Verify a round-trip**

Run:
```bash
.venv/bin/python -c "from datasets import load_dataset; d=load_dataset('udonpred/datasets','trizod'); print({k: d[k].num_rows for k in d})"
```
Expected: a dict with `train`/`validation`/`test` row counts.

- [ ] **Step 3: Update `README.md`**

Add a "Datasets" subsection under Retraining documenting:
`load_dataset("udonpred/datasets", "<target>")`; that embeddings live at
`<target>/embeddings/<plm>/<split>.h5`; `udonpred-publish-dataset`;
`udonpred-embed --push-to-hub --dataset <target> --split <split>`; and the
`--embeddings udonpred/datasets:<target>/embeddings/<plm>/<split>.h5` inference
reference. Note that generating embeddings needs the `[embedding]` stack + GPU.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: document Hub datasets and per-pLM embedding distribution

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-review

- **Spec coverage:** dataset repo + configs (Tasks 3, 7); embeddings-in-repo subfolder + `plm_slug` (Tasks 1, 3); precomputed-first training (Tasks 5, 6); Hub-ref `--embeddings` (Task 2); publish tooling incl. `udonpred-embed --push-to-hub` (Tasks 3, 4); lean/torch-free + boundary guard (Tasks 1, 2); one-time upload + deferred embeddings (Task 7); KISS config (Task 6). trizod2 excluded (Task 3 `EXCLUDED_TARGETS`). All covered.
- **Placeholder scan:** none — every code step has concrete content.
- **Type consistency:** `resolve_embeddings(target, split, plm, …)`, `resolve_embeddings_ref(source, revision=None)`, `publish_embeddings(h5_path, target, split, plm, repo=None)`, `attach_precomputed_embeddings(dataset_split, h5_path, id_column, column)`, `plm_slug(name)` used consistently across tasks.
