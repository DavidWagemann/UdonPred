"""Generate ProstT5 per-residue embeddings for UdonPred / CAID4.

This is the *exact command* the CAID4 organizers (or anyone running offline)
should use to precompute the embeddings consumed by
:mod:`udonpred.caid.predict`. It is the only component that loads the ProstT5
protein language model, so it lives outside the CAID container.

Output formats:
  * ``.h5``  — one dataset per sequence, keyed by the FASTA header
               (use this for multi-sequence FASTA files).
  * ``.npy`` — a single ``(L, 1024)`` array (single-sequence FASTA only).

Embeddings are written already trimmed to one row per residue (the ProstT5
``<AA2fold>`` prefix and trailing EOS are removed), shape ``(L, 1024)``,
float32. Ambiguous residues (B, Z, J, U, O, ``*``) are mapped to ``X`` before
embedding.

Example::

    udonpred-embed input.fasta --output embeddings.h5 --device cpu
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from udonpred.embedding.backbone import (
    BACKBONE_NAME,
    compute_embeddings,
    load_backbone,
    load_tokenizer,
    resolve_device,
    tokenize_batch,
)
from udonpred.fasta import read_fasta
from udonpred.inference import count_batches, iter_batches


def generate(
    fasta: str,
    output: str,
    device: str,
    max_total_seq_len: int,
) -> None:
    entries = read_fasta(fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    out_path = Path(output)
    suffix = out_path.suffix.lower()
    if suffix not in (".h5", ".hdf5", ".npy"):
        raise ValueError(f"Output must end in .h5 or .npy, got {suffix!r}.")
    if suffix == ".npy" and len(entries) > 1:
        raise ValueError(
            ".npy output supports a single sequence only; use .h5 for "
            f"{len(entries)} sequences."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import torch

    torch_device = resolve_device(device)
    torch_dtype = torch.float16 if torch_device == "cuda" else torch.float32

    print(f"Loading tokenizer ({BACKBONE_NAME}) ...")
    tokenizer = load_tokenizer()
    print(f"Loading backbone ({BACKBONE_NAME}) ...")
    backbone = load_backbone(torch_device, torch_dtype)

    results: dict[str, np.ndarray] = {}
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
            for header, seq, emb in zip(headers, seqs, emb_np):
                results[header] = emb[: len(seq)].astype(np.float32)

    if suffix == ".npy":
        np.save(out_path, next(iter(results.values())))
    else:
        import h5py

        with h5py.File(out_path, "w") as f:
            for header, emb in results.items():
                f.create_dataset(header, data=emb)
    print(f"Wrote embeddings for {len(results)} sequence(s) -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ProstT5 per-residue embeddings for UdonPred."
    )
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output file path (.h5 keyed by header, or .npy single sequence).",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for embedding (auto, cpu, cuda).",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=2000,
        help="Max total sequence length per batch (dynamic batching).",
    )
    args = parser.parse_args()
    generate(args.fasta, args.output, args.device, args.batch_size)


if __name__ == "__main__":
    main()
