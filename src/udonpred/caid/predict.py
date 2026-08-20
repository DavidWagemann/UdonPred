"""CAID4 command-line runner for UdonPred (precomputed-embedding inference).

This is the CAID-compliant entry point: it does **not** load or run the
ProstT5 protein language model. Per-residue embeddings are supplied externally
(``.npy`` or ``.h5``, see ``udonpred-embed`` / EMBEDDINGS.md for how to
generate them) and only the small ONNX prediction heads are run, on CPU by
default.

Output follows the CAID layout: one directory per prediction head ("flavor"),
holding one file per protein — ``{output}/{target}/{protein}.caid``. Each row is
``<index>\\t<residue>\\t<score>\\t<binary>``.

Both output conventions come from ``udonpred.inference.TARGET_POLICIES``: the
binary column thresholds each head on its own raw scale, and by default
(``--normalize``) the score column is mapped onto ``[0, 1]`` with higher meaning
more disordered, which flips the ``chezod`` and ``plddt`` heads and rescales
every non-sigmoid head. Pass ``--no-normalize`` to keep raw scores; the binary
column is unaffected either way.

Example::

    udonpred-caid input.fasta weights/ \\
        --embeddings embeddings.h5 --target all --threads 24 --output out/
"""

import argparse

from udonpred.cli import common_parser
from udonpred.datasets import resolve_embeddings_ref
from udonpred.fasta import read_fasta
from udonpred.heads import resolve_model_dir, resolve_targets
from udonpred.inference import load_head, score_embeddings, unknown_targets
from udonpred.output import CaidWriter

from udonpred.caid.embeddings import align_embedding, load_precomputed_embeddings


def run(
    fasta: str,
    model_dir: str | None,
    embeddings: str,
    target: list[str],
    output_path: str | None,
    device: str,
    threads: int | None,
    smooth: float,
    revision: str | None = None,
    normalize: bool = True,
) -> None:
    entries = read_fasta(fasta)
    if not entries:
        raise ValueError("No FASTA entries found.")

    # model_dir may be a local directory (used as-is, offline) or a Hub repo id;
    # omitting it pulls the pinned release from the Hub.
    model_dir_path = resolve_model_dir(model_dir, revision)

    targets = resolve_targets(model_dir_path, target)

    # Fail before loading any head or embedding file if a requested head has no
    # registered policy — its threshold is needed for the binary column, and its
    # scale for --normalize.
    unknown = unknown_targets(targets)
    if unknown:
        raise ValueError(
            f"No policy registered for: {', '.join(unknown)}. Add it to "
            "udonpred.inference.TARGET_POLICIES."
        )

    # `embeddings` may be a local .h5/.npy path or a Hub reference
    # (repo_id:path_in_repo[@revision]).
    source = load_precomputed_embeddings(str(resolve_embeddings_ref(embeddings)))
    # Load every requested head once; the embedding is computed/aligned once per
    # sequence and reused across all heads.
    heads = {
        name: load_head(model_dir_path / f"{name}.onnx", device, threads=threads)
        for name in targets
    }

    writer = CaidWriter(output_path, targets, smooth=smooth, normalize=normalize)

    # The embedding is aligned once per sequence and reused across every head.
    for index, (header, seq) in enumerate(entries):
        emb = align_embedding(source.get(header, index), len(seq))
        batched = emb[None, ...]
        for name, head in heads.items():
            scores_seq = score_embeddings(head, batched)[0][: len(seq)]
            writer.write(name, header, seq, scores_seq)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "CAID4 UdonPred runner: predict protein disorder from precomputed "
            "ProstT5 embeddings (no PLM, CPU-only by default)."
        ),
        parents=[common_parser(device_choices=["cpu", "cuda"], device_default="cpu")],
    )
    parser.add_argument(
        "--embeddings",
        "-e",
        type=str,
        required=True,
        help="Path to precomputed ProstT5 embeddings (.npy single sequence, "
        "or .h5 keyed by FASTA header), or a Hub reference like "
        "udonpred/datasets:trizod/embeddings/prostt5/test.h5.",
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
        revision=args.revision,
        normalize=args.normalize,
    )


if __name__ == "__main__":
    main()
