"""Standard UdonPred runner: embeds sequences with ProstT5 on the fly.

This is the original, self-contained predictor — it loads the ProstT5 backbone
and computes embeddings itself. For the CAID4-compliant runner that consumes
precomputed embeddings (no PLM in the container), see
:mod:`udonpred.caid.predict`.

Both runners share the FASTA, ONNX-head, and embedding logic in the
``udonpred`` package; this script only adds the on-the-fly embedding loop.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm

from udonpred.embedding.backbone import (
    BACKBONE_NAME,
    compute_embeddings,
    load_backbone,
    load_tokenizer,
    resolve_device,
    tokenize_batch,
)
from udonpred.fasta import format_predictions, read_fasta
from udonpred.heads import (
    DEFAULT_HEADS_REPO,
    DEFAULT_HEADS_REVISION,
    resolve_model_dir,
)
from udonpred.inference import (
    count_batches,
    iter_batches,
    load_head,
    score_embeddings,
    smooth_scores,
)


def run_exported(
    entries: List[Tuple[str, str]],
    model_dir: str,
    target: str,
    max_total_seq_len: int,
    output_path: str | None,
    device: str,
    smooth: float = 1.5,
    threads: int | None = None,
) -> None:
    """Run prediction using the HuggingFace backbone and ONNX prediction head."""
    torch_device = resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"Loading tokenizer ({BACKBONE_NAME}) ...")
    tokenizer = load_tokenizer()

    print(f"Loading backbone ({BACKBONE_NAME}) ...")
    backbone = load_backbone(torch_device, torch_dtype)

    onnx_path = Path(model_dir) / f"{target}.onnx"
    print(f"Loading head ({onnx_path}) ...")
    head = load_head(onnx_path, torch_device, threads=threads)

    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    total_batches = count_batches(entries, max_total_seq_len)
    with torch.inference_mode():
        for batch in tqdm(
            iter_batches(entries, max_total_seq_len), total=total_batches
        ):
            headers, seqs = zip(*batch)
            seqs = list(seqs)
            max_seq_len = max(len(s) for s in seqs)

            input_ids, attention_mask = tokenize_batch(tokenizer, seqs, torch_device)
            emb_np = compute_embeddings(
                backbone, input_ids, attention_mask, max_seq_len
            )
            scores_batch = score_embeddings(head, emb_np)

            for header, seq, scores in zip(headers, seqs, scores_batch):
                scores_seq = smooth_scores(scores[: len(seq)], smooth)
                formatted_lines = format_predictions(header, seq, scores_seq)
                if out_dir:
                    safe_header = header.replace("/", "_").replace("|", "_")
                    file_path = out_dir / f"udonpred_{safe_header}.caid"
                    with file_path.open("w") as f:
                        f.writelines(formatted_lines)
                else:
                    sys.stdout.writelines(formatted_lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict different definitions of protein disorder."
    )
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "model_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing the ONNX heads, or a Hugging Face repo id. "
        f"A local directory is used as-is (offline). Omit to pull the pinned "
        f"Hub release ({DEFAULT_HEADS_REPO} @ {DEFAULT_HEADS_REVISION}).",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        default="trizod",
        help="Prediction type — must match a {target}.onnx file in model_dir",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hub revision (tag/branch/commit) to pull heads from when model_dir "
        f"is a repo id or omitted (default: {DEFAULT_HEADS_REVISION}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory path (will save each sequence as a .caid file)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2000,
        help="Max total sequence length per batch (dynamic batching)",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for inference (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--threads",
        "-j",
        type=int,
        default=None,
        help="Max CPU threads for ONNX inference (default: onnxruntime decides).",
    )
    parser.add_argument(
        "--smooth",
        "-s",
        type=float,
        default=1.5,
        metavar="SIGMA",
        help="Sigma for Gaussian smoothing applied to per-residue scores "
        "(0 to disable, default: 1.5)",
    )

    args = parser.parse_args()

    entries = read_fasta(args.fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    entries = sorted(entries, key=lambda item: len(item[1]), reverse=True)

    # Local directory (used as-is) or a Hub repo id; omitting it pulls the
    # pinned release from the Hub.
    model_dir = str(resolve_model_dir(args.model_dir, args.revision))

    run_exported(
        entries,
        model_dir,
        args.target,
        args.batch_size,
        args.output,
        args.device,
        args.smooth,
        args.threads,
    )


if __name__ == "__main__":
    main()
