"""FASTA reading and sequence sanitization."""

import re
from typing import List, Tuple

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
