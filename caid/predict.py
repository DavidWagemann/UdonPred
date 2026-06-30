"""CAID4 command-line runner for UdonPred (precomputed-embedding inference).

This is the CAID-compliant entry point: it does **not** load or run the
ProstT5 protein language model. Per-residue embeddings are supplied externally
(``.npy`` or ``.h5``, see ``embed.py`` / EMBEDDINGS.md for how to generate
them) and only the small ONNX prediction heads are run, on CPU by default.

Each prediction head writes a single ``udonpred_{target}.caid`` file containing all
input proteins concatenated; with ``--target all`` every head's file lands flat
in the same output directory.

Example::

    python -m caid.predict input.fasta weights/ \\
        --embeddings embeddings.h5 --target all --threads 24 --output out/
"""

import argparse
import sys
from pathlib import Path

from udonpred.fasta import format_predictions, read_fasta
from udonpred.inference import load_head, score_embeddings, smooth_scores

from caid.embeddings import align_embedding, load_precomputed_embeddings


def discover_targets(model_dir) -> list[str]:
    """Return the sorted stem names of every ``*.onnx`` head in ``model_dir``."""
    return sorted(p.stem for p in Path(model_dir).glob("*.onnx"))


def resolve_targets(model_dir, requested: list[str]) -> list[str]:
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


def run(
    fasta: str,
    model_dir: str,
    embeddings: str,
    target: list[str],
    output_path: str | None,
    device: str,
    threads: int | None,
    smooth: float,
) -> None:
    entries = read_fasta(fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    model_dir_path = Path(model_dir)
    if not model_dir_path.is_dir():
        raise ValueError(f"Model directory not found: {model_dir}")

    targets = resolve_targets(model_dir_path, target)
    multi = len(targets) > 1

    source = load_precomputed_embeddings(embeddings)
    # Load every requested head once; the embedding is computed/aligned once per
    # sequence and reused across all heads.
    heads = {
        name: load_head(model_dir_path / f"{name}.onnx", device, threads=threads)
        for name in targets
    }

    # One CAID file per prediction head, holding all proteins concatenated, all
    # written flat into the output directory (udonpred_{target}.caid). Open every head's
    # file once so each sequence's embedding is aligned a single time and reused.
    out_files = None
    if output_path:
        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_files = {name: (out_dir / f"udonpred_{name}.caid").open("w") for name in targets}

    try:
        for index, (header, seq) in enumerate(entries):
            emb = align_embedding(source.get(header, index), len(seq))
            batched = emb[None, ...]
            for name, head in heads.items():
                scores_seq = score_embeddings(head, batched)[0][: len(seq)]
                scores_seq = smooth_scores(scores_seq, smooth)
                formatted_lines = format_predictions(header, seq, scores_seq)

                if out_files is not None:
                    out_files[name].writelines(formatted_lines)
                else:
                    if multi:
                        sys.stdout.write(f"# target: {name}\n")
                    sys.stdout.writelines(formatted_lines)
    finally:
        if out_files is not None:
            for handle in out_files.values():
                handle.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CAID4 UdonPred runner: predict protein disorder from precomputed "
            "ProstT5 embeddings (no PLM, CPU-only by default)."
        )
    )
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "model_dir", type=str, help="Directory containing ONNX head files"
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=str,
        required=True,
        help="Path to precomputed ProstT5 embeddings (.npy single sequence, "
        "or .h5 keyed by FASTA header).",
    )
    parser.add_argument(
        "--target",
        "-t",
        type=str,
        nargs="+",
        default=["trizod"],
        metavar="TARGET",
        help="One or more prediction types, each matching a {target}.onnx file "
        "in model_dir (trizod, chezod, softdis, pdbflex, atlas, plddt, "
        "disprot). Use 'all' to run every head in model_dir. Each head writes "
        "one udonpred_{target}.caid file (all proteins concatenated) into the output "
        "directory. Default: trizod.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory. Each head writes one udonpred_<target>.caid file with "
        "all proteins concatenated. Writes to stdout if unset.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for ONNX inference (default: cpu).",
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
        help="Sigma for Gaussian smoothing of per-residue scores "
        "(0 to disable, default: 1.5)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run(
        fasta=args.fasta,
        model_dir=args.model_dir,
        embeddings=args.embeddings,
        target=args.target,
        output_path=args.output,
        device=args.device,
        threads=args.threads,
        smooth=args.smooth,
    )


if __name__ == "__main__":
    main()
