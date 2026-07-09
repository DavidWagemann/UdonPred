"""Evaluate the old trizod head against the retrained trizod2 head on the trizod test set.

Both heads map per-residue ProstT5 embeddings -> per-residue TriZOD scores. We
score the trizod *test* split (labels from the local jsonl; embeddings pulled
from udonpred/datasets, keyed by the jsonl id), mask missing residues (y == 999),
pool over all residues, and report MAE / MSE / Spearman / Pearson — the same
pooled, 999-masked convention the training metrics use.

No pLM / torch needed: only onnxruntime + the precomputed embeddings.

    uv run --extra hub python eval_trizod_heads.py
    uv run --extra hub python eval_trizod_heads.py --smooth 1.5   # deployed post-proc
    uv run --extra hub python eval_trizod_heads.py --min-length 50  # long proteins only
    # override a head with a local file or a repo:file@revision reference:
    uv run --extra hub python eval_trizod_heads.py --new weights/trizod2.onnx
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

from udonpred.filtering import filter_rows_min_length
from udonpred.inference import load_head, score_embeddings, smooth_scores

MASK_VALUE = 999
TEST_JSONL = "data/split/trizod/test.jsonl"
EMB_REPO = "udonpred/datasets"
EMB_FILE = "trizod/embeddings/prostt5/test.h5"

# Default heads to compare (both from the Hub for reproducibility):
#   old  = the pre-update trizod head (release v0.1.0)
#   new  = the retrained trizod2 head (release v0.2.0)
DEFAULT_OLD = "udonpred/prediction-heads:trizod.onnx@v0.1.0"
DEFAULT_NEW = "udonpred/prediction-heads:trizod2.onnx@v0.2.0"


def resolve_head(spec: str) -> Path:
    """A local path, or a ``repo_id:file.onnx[@revision]`` Hub (model) reference."""
    if os.path.exists(spec):
        return Path(spec)
    repo_id, _, rest = spec.partition(":")
    if "/" not in repo_id or not rest:
        raise ValueError(f"Not a local file or repo:file reference: {spec!r}")
    filename, _, revision = rest.partition("@")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id, filename, revision=revision or None, repo_type="model")
    )


def predictions_and_labels(head, rows, emb_h5, smooth):
    """Return pooled (predictions, labels) over all unmasked residues."""
    import h5py

    preds, labels = [], []
    with h5py.File(emb_h5, "r") as f:
        for row in rows:
            y = np.asarray(row["y"], dtype=np.float32)  # (L,), 999 = missing
            emb = np.asarray(f[str(row["id"])], dtype=np.float32)  # (L, 1024)
            scores = score_embeddings(head, emb[None, ...])[0].reshape(-1)[: len(y)]
            scores = smooth_scores(scores, smooth)
            keep = y != MASK_VALUE
            preds.append(scores[keep])
            labels.append(y[keep])
    return np.concatenate(preds), np.concatenate(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", default=DEFAULT_OLD, help=f"Old head (default: {DEFAULT_OLD}).")
    parser.add_argument("--new", default=DEFAULT_NEW, help=f"New head (default: {DEFAULT_NEW}).")
    parser.add_argument("--smooth", type=float, default=0.0,
                        help="Gaussian sigma for post-processing (0 = raw, matches "
                        "training metrics; 1.5 = the deployed predictor default).")
    parser.add_argument("--min-length", type=int, default=0,
                        help="Only score proteins with at least this many residues "
                        "(0 = all). Matches the training-side min_length filter.")
    args = parser.parse_args()

    with open(TEST_JSONL) as f:
        rows = [json.loads(line) for line in f]

    n_total = len(rows)
    rows = filter_rows_min_length(rows, args.min_length)

    from huggingface_hub import hf_hub_download

    emb_h5 = hf_hub_download(EMB_REPO, EMB_FILE, repo_type="dataset")

    kept = f"{len(rows)}/{n_total}" if args.min_length else str(len(rows))
    length_note = f", min_length = {args.min_length}" if args.min_length else ""
    print(f"trizod test: {kept} proteins  (smooth sigma = {args.smooth}{length_note})")
    print(f"  old = {args.old}")
    print(f"  new = {args.new}")
    header = f"{'head':<5s} {'MAE':>8s} {'MSE':>8s} {'Spearman':>9s} {'Pearson':>8s} {'n_res':>8s}"
    print(header)
    print("-" * len(header))
    for name, spec in (("old", args.old), ("new", args.new)):
        head = load_head(resolve_head(spec), "cpu")
        pred, y = predictions_and_labels(head, rows, emb_h5, args.smooth)
        mae = float(np.mean(np.abs(pred - y)))
        mse = float(np.mean((pred - y) ** 2))
        rho = float(spearmanr(pred, y).statistic)
        r = float(pearsonr(pred, y).statistic)
        print(f"{name:<5s} {mae:8.4f} {mse:8.4f} {rho:9.4f} {r:8.4f} {len(y):8d}")


if __name__ == "__main__":
    main()
