"""Upload UdonPred datasets and per-pLM embeddings to the Hub.

Publishes the per-target train/valid/test FASTA+jsonl into a single dataset
repo (with a per-target ``configs:`` card) and uploads precomputed embeddings
under ``<target>/embeddings/<plm>/<split>.h5``. Uses the ambient
``huggingface-cli login``; ``huggingface_hub`` is imported lazily.
"""

import argparse
from pathlib import Path

from udonpred.datasets import DEFAULT_DATASET_REPO

# Subdirectories to never upload as datasets (none currently). Hidden dirs and
# dirs without a train.jsonl are skipped automatically by _discover_targets.
EXCLUDED_TARGETS: set = set()
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
