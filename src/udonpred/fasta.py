"""FASTA reading, sequence sanitisation, and CAID output formatting."""

import re
from typing import List, Sequence, Tuple

# Ambiguous / non-standard residues that the ProstT5 vocabulary does not cover.
# All are mapped to ``X`` (any amino acid):
#   B -> Asn/Asp, Z -> Gln/Glu, J -> Leu/Ile, U -> Selenocysteine,
#   O -> Pyrrolysine, ``*`` -> stop/translation artifact.
# ``X`` itself is already valid and is left untouched.
_NONSTANDARD = re.compile(r"[BZJUO*]")


def read_fasta(path: str) -> List[Tuple[str, str]]:
    """Read a FASTA file and return a list of ``(header, sequence)`` tuples.

    Internal whitespace and blank lines are stripped; headers keep everything
    after the leading ``>`` (trimmed).
    """
    entries: List[Tuple[str, str]] = []
    header = None
    seq_chunks: List[str] = []

    with open(path, "r") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    entries.append((header, "".join(seq_chunks)))
                header = line[1:].strip()
                seq_chunks = []
            else:
                seq_chunks.append(re.sub(r"\s+", "", line))

    if header is not None:
        entries.append((header, "".join(seq_chunks)))

    return entries


def sanitize_sequence(seq: str) -> str:
    """Replace ambiguous/non-standard residues (B, Z, J, U, O, ``*``) with ``X``."""
    return _NONSTANDARD.sub("X", seq)


def _unwrap(row):
    """Take the scalar out of a ``(1,)``-shaped per-residue row."""
    if hasattr(row, "__len__") and not isinstance(row, str):
        return row[0] if len(row) > 0 else 0.0
    return row


def safe_filename(header: str) -> str:
    """Turn a FASTA header into a filename-safe stem.

    Path separators, pipes, and whitespace are replaced rather than stripped, so
    no two distinct headers collapse onto the same file.
    """
    return re.sub(r"\s+", "_", header.replace("/", "_").replace("|", "_")).strip("_")


def format_predictions(
    header: str,
    sequence: str,
    scores: Sequence,
    binary: Sequence | None = None,
) -> List[str]:
    """Format per-residue scores into CAID ``.caid`` lines.

    Each line is ``<index>\t<residue>\t<score:.3f>\t<binary>``, preceded by a
    ``>header`` line. Omitting ``binary`` leaves the fourth column empty.
    """
    lines: List[str] = [f">{header}\n"]
    for idx, (aa, row) in enumerate(zip(sequence, scores), start=1):
        call = "" if binary is None else str(int(_unwrap(binary[idx - 1])))
        lines.append(f"{idx}\t{aa}\t{float(_unwrap(row)):.3f}\t{call}\n")
    return lines
