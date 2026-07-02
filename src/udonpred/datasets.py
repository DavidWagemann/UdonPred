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
