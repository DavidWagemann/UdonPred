"""Resolve UdonPred ONNX prediction heads from a local directory or the Hub.

The heads live in the public Hugging Face repo ``udonpred/prediction-heads``
(the seven ``*.onnx`` files of the pinned release, tagged per release). Inference can consume them
either from a local directory (e.g. the weights baked into the CAID container)
or by downloading a pinned revision from the Hub.

Only the download path imports :mod:`huggingface_hub`, so purely-local use --
the CAID container passing its baked-in ``weights/`` -- stays hub-free and needs
no network access. Publishing new heads after (re)training lives in
:mod:`udonpred.utils.export`.
"""

import os
from pathlib import Path

# Source of truth for the released prediction heads.
DEFAULT_HEADS_REPO = "udonpred/prediction-heads"
# Pinned, overridable: bump this (and publish a matching tag) on retraining.
# v0.1.0 is the CAID-submission set — the seven established heads. v0.2.0 adds
# trizod2 (the other seven blobs are identical), which `--target all` would
# otherwise pick up; pass --revision v0.2.0 or a repo id to opt into it.
DEFAULT_HEADS_REVISION = "v0.1.0"

# Environment overrides (all optional).
ENV_HEADS_DIR = "UDONPRED_HEADS_DIR"
ENV_HEADS_REPO = "UDONPRED_HEADS_REPO"
ENV_HEADS_REVISION = "UDONPRED_HEADS_REVISION"


def resolve_model_dir(source: str | None = None, revision: str | None = None) -> Path:
    """Return a local directory containing the ONNX prediction heads.

    Resolution order:

    1. ``source`` names an existing local directory -> used as-is (no Hub import,
       no network).
    2. ``source`` is ``None`` and ``$UDONPRED_HEADS_DIR`` points at an existing
       directory -> used as-is.
    3. Otherwise ``source`` (or, when ``None``, ``$UDONPRED_HEADS_REPO`` or
       :data:`DEFAULT_HEADS_REPO`) is treated as a Hugging Face repo id and its
       ``*.onnx`` files are downloaded (cached) at ``revision``
       (or ``$UDONPRED_HEADS_REVISION`` / :data:`DEFAULT_HEADS_REVISION`).

    Args:
        source: A local directory path *or* a Hugging Face repo id. ``None``
            selects the default local dir / repo via the resolution order above.
        revision: Hub revision (tag, branch, or commit) for the download path.

    Returns:
        Path to a directory containing the ``*.onnx`` heads.
    """
    # 1. explicit local directory
    if source is not None:
        candidate = Path(source)
        if candidate.is_dir():
            return candidate
    else:
        # 2. environment-provided local directory
        env_dir = os.environ.get(ENV_HEADS_DIR)
        if env_dir and Path(env_dir).is_dir():
            return Path(env_dir)

    # 3. treat as a Hub repo id and download
    repo_id = source or os.environ.get(ENV_HEADS_REPO) or DEFAULT_HEADS_REPO
    rev = revision or os.environ.get(ENV_HEADS_REVISION) or DEFAULT_HEADS_REVISION
    return _download_heads(repo_id, rev)


def discover_targets(model_dir: str | Path) -> list[str]:
    """Return the sorted stem names of every ``*.onnx`` head in ``model_dir``."""
    return sorted(p.stem for p in Path(model_dir).glob("*.onnx"))


def resolve_targets(model_dir: str | Path, requested: list[str]) -> list[str]:
    """Resolve requested target names to validated head stems.

    ``["all"]`` expands to every ``*.onnx`` head in ``model_dir``; otherwise
    each requested name must have a matching ``{name}.onnx`` file.
    """
    model_dir_path = Path(model_dir)
    if requested == ["all"]:
        targets = discover_targets(model_dir_path)
        if not targets:
            raise ValueError(f"No .onnx heads found in {model_dir}")
        return targets
    for name in requested:
        if not (model_dir_path / f"{name}.onnx").exists():
            raise ValueError(f"ONNX head not found: {model_dir_path / f'{name}.onnx'}")
    return requested


def _download_heads(repo_id: str, revision: str) -> Path:
    """Download the ``*.onnx`` heads from the Hub, returning the cached directory."""
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:  # pragma: no cover - trivial guard
        raise ModuleNotFoundError(
            f"Fetching prediction heads from the Hub ({repo_id!r}) requires "
            "huggingface_hub. Install it with `pip install 'udonpred-comp[hub]'`, "
            "or pass a local directory containing the *.onnx heads instead."
        ) from exc
    local = snapshot_download(
        repo_id, revision=revision, allow_patterns=["*.onnx"]
    )
    return Path(local)
