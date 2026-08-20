"""Standard UdonPred runner: embeds sequences with ProstT5 on the fly.

This is the original, self-contained predictor — it loads the ProstT5 backbone
and computes embeddings itself. For the CAID4-compliant runner that consumes
precomputed embeddings (no PLM in the container), see
:mod:`udonpred.caid.predict`.

Both runners share the FASTA, ONNX-head, post-processing, output, and CLI logic
in the ``udonpred`` package, and write the identical CAID layout: one directory
per prediction head, holding one ``{protein}.caid`` file per input protein, whose
rows are ``<index>\\t<residue>\\t<score>\\t<binary>``. This script only adds the
on-the-fly embedding loop.
"""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from tqdm import tqdm

from udonpred.cli import common_parser
from udonpred.embedding.backbone import (
    BACKBONE_NAME,
    compute_embeddings,
    load_backbone,
    load_tokenizer,
    resolve_device,
    tokenize_batch,
)
from udonpred.fasta import read_fasta
from udonpred.heads import resolve_model_dir, resolve_targets
from udonpred.inference import (
    count_batches,
    iter_batches,
    load_head,
    score_embeddings,
    unknown_targets,
)
from udonpred.output import CaidWriter


def run_exported(
    entries: list[tuple[str, str]],
    model_dir: str,
    target: list[str],
    max_total_seq_len: int,
    output_path: str | None,
    device: str,
    smooth: float = 1.5,
    threads: int | None = None,
    normalize: bool = True,
) -> None:
    """Run prediction using the HuggingFace backbone and ONNX prediction heads."""
    targets = resolve_targets(model_dir, target)

    # Fail before loading the backbone if a requested head has no registered
    # policy — its threshold is needed for the binary column, and its scale for
    # --normalize.
    unknown = unknown_targets(targets)
    if unknown:
        raise ValueError(
            f"No policy registered for: {', '.join(unknown)}. Add it to "
            "udonpred.inference.TARGET_POLICIES."
        )

    torch_device = resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"Loading tokenizer ({BACKBONE_NAME}) ...")
    tokenizer = load_tokenizer()

    print(f"Loading backbone ({BACKBONE_NAME}) ...")
    backbone = load_backbone(torch_device, torch_dtype)

    # Load every requested head once; each batch is embedded a single time and
    # the embeddings reused across all heads.
    heads = {}
    for name in targets:
        onnx_path = Path(model_dir) / f"{name}.onnx"
        print(f"Loading head ({onnx_path}) ...")
        heads[name] = load_head(onnx_path, torch_device, threads=threads)

    total_batches = count_batches(entries, max_total_seq_len)
    with CaidWriter(
        output_path, targets, smooth=smooth, normalize=normalize
    ) as writer, torch.inference_mode():
        for batch in tqdm(
            iter_batches(entries, max_total_seq_len), total=total_batches
        ):
            headers, seqs = zip(*batch)
            seqs = list(seqs)
            max_seq_len = max(len(s) for s in seqs)

            # One tokenize+embed pass serves the whole batch and every head, so
            # its cost is split per protein and charged to each head's timing.
            embed_start = perf_counter()
            input_ids, attention_mask = tokenize_batch(tokenizer, seqs, torch_device)
            emb_np = compute_embeddings(
                backbone, input_ids, attention_mask, max_seq_len
            )
            embed_ms = (perf_counter() - embed_start) * 1000.0 / len(seqs)

            for name, head in heads.items():
                scores_batch = score_embeddings(head, emb_np)
                for header, seq, scores in zip(headers, seqs, scores_batch):
                    with writer.timing(name, header, extra_ms=embed_ms):
                        writer.write(name, header, seq, scores[: len(seq)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict different definitions of protein disorder.",
        parents=[
            common_parser(
                device_choices=["auto", "cpu", "cuda"], device_default="auto"
            )
        ],
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2000,
        help="Max total sequence length per batch (dynamic batching)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

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
        args.normalize,
    )


if __name__ == "__main__":
    main()
