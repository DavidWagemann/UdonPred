"""Argument definitions shared by the two prediction runners.

:mod:`udonpred.caid.predict` and :mod:`udonpred.embedding.predict` take the same
inputs and produce the same CAID output, differing only in where the embeddings
come from. The flags describing that shared contract are defined here once and
pulled in as a parent parser, so the two CLIs cannot drift apart.
"""

import argparse

from udonpred.heads import DEFAULT_HEADS_REPO, DEFAULT_HEADS_REVISION


def common_parser(
    device_choices: list[str],
    device_default: str,
) -> argparse.ArgumentParser:
    """Build the parent parser holding every flag both runners share.

    Args:
        device_choices: Accepted ``--device`` values. The CAID runner is
            CPU/CUDA only, while the on-the-fly runner also accepts ``auto``.
        device_default: Default ``--device`` value.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("fasta", type=str, help="Path to input FASTA file")
    parser.add_argument(
        "model_dir",
        type=str,
        nargs="?",
        default=None,
        help="Directory containing the ONNX heads, or a Hugging Face repo id. "
        "A local directory is used as-is (offline). Omit to pull the pinned "
        f"Hub release ({DEFAULT_HEADS_REPO} @ {DEFAULT_HEADS_REVISION}).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Hub revision (tag/branch/commit) to pull heads from when model_dir "
        f"is a repo id or omitted (default: {DEFAULT_HEADS_REVISION}).",
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
        "its own {target}/ subdirectory of the output directory. Default: trizod.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory. Each head gets a {target}/ subdirectory holding "
        "one {protein}.caid file per input protein. Writes to stdout if unset.",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default=device_default,
        choices=device_choices,
        help=f"Device for inference ({', '.join(device_choices)}; "
        f"default: {device_default}).",
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
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Map scores onto the CAID convention — [0, 1] where higher means "
        "more disordered: flip chezod/plddt and rescale every non-sigmoid head "
        "by its fixed scale. Does not affect the binary column. Use "
        "--no-normalize for raw head output (default: enabled).",
    )
    return parser
