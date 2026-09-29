"""Shared per-residue post-processing, ``.caid`` formatting, and output writing.

Every runner turns a head's raw output into the same CAID artifacts, so that
chain lives here rather than in any one runner. Nothing in this module imports
``torch``, keeping the lean core torch-free.
"""

import csv
import re
import sys
import time
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from udonpred.inference import binarize_scores, normalize_scores, smooth_scores

# Banner line opening every timings.csv. Deliberately carries no timestamp, so
# repeated runs over the same input produce byte-identical output.
TIMINGS_BANNER = "# Running UdonPred"


def postprocess_scores(
    scores: np.ndarray,
    target: str,
    smooth: float,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one head's raw output into the two CAID columns.

    Smoothing runs first, on the head's own scale, so the binary thresholds --
    which are expressed in raw units -- are read off the smoothed-but-unscaled
    scores. Only then is the score column rescaled, which is why the binary
    calls are identical whether or not ``normalize`` is set.

    Returns:
        ``(scores, binary)``, both shaped like the input.
    """
    scores = smooth_scores(scores, smooth)
    binary = binarize_scores(scores, target)
    if normalize:
        scores = normalize_scores(scores, target)
    return scores, binary


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
) -> list[str]:
    """Format per-residue scores into CAID ``.caid`` lines.

    Each line is ``<index>\t<residue>\t<score:.3f>\t<binary>``, preceded by a
    ``>header`` line. Omitting ``binary`` leaves the fourth column empty.
    """
    lines: list[str] = [f">{header}\n"]
    for idx, (aa, row) in enumerate(zip(sequence, scores), start=1):
        call = "" if binary is None else str(int(_unwrap(binary[idx - 1])))
        lines.append(f"{idx}\t{aa}\t{float(_unwrap(row)):.3f}\t{call}\n")
    return lines


class CaidWriter:
    """Write predictions in the CAID layout: ``{output}/{target}/{protein}.caid``.

    One directory per prediction head (the CAID "flavor"), holding one file per
    protein. Without an output directory everything goes to stdout instead,
    prefixed by a ``# target: <name>`` line when more than one head is running.

    Smoothing and normalization are run-wide settings, so they are fixed here
    once and applied to every :meth:`write` call.

    Each flavor directory also gets a ``timings.csv`` recording how long that
    head took per protein, written on :meth:`close` (so use this as a context
    manager). Its banner carries no timestamp, so re-running over the same input
    yields byte-identical files. Timings are collected by wrapping the per-protein work in
    :meth:`timing`; nothing is written in stdout mode, which has no directories
    to put the file in.
    """

    def __init__(
        self,
        output_path: str | None,
        targets: list[str],
        smooth: float = 1.5,
        normalize: bool = True,
    ) -> None:
        self.smooth = smooth
        self.normalize = normalize
        self._multi = len(targets) > 1
        self._timings: dict[str, list[tuple[str, float]]] = {n: [] for n in targets}
        self._dirs: dict[str, Path] | None = None
        if output_path:
            self._dirs = {}
            for name in targets:
                directory = Path(output_path) / name
                directory.mkdir(parents=True, exist_ok=True)
                self._dirs[name] = directory

    def write(
        self,
        target: str,
        header: str,
        sequence: str,
        raw_scores: np.ndarray,
    ) -> None:
        """Post-process one protein's raw scores for one head and emit them."""
        scores, binary = postprocess_scores(
            raw_scores, target, self.smooth, self.normalize
        )
        lines = format_predictions(header, sequence, scores, binary)
        if self._dirs is None:
            if self._multi:
                sys.stdout.write(f"# target: {target}\n")
            sys.stdout.writelines(lines)
            return
        out_file = self._dirs[target] / f"{safe_filename(header)}.caid"
        with out_file.open("w") as handle:
            handle.writelines(lines)

    @contextmanager
    def timing(self, target: str, header: str, extra_ms: float = 0.0):
        """Record how long the wrapped per-protein work took for one head.

        Args:
            target: The head the work belongs to.
            header: The protein, as it appears in the FASTA.
            extra_ms: Milliseconds of shared preparation to attribute to this
                head -- embedding alignment (or computation) is done once per
                protein and reused across heads, but a run of this head alone
                would still have paid it, so each head is charged for it.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self._timings[target].append((header, elapsed_ms + extra_ms))

    def close(self) -> None:
        """Write each flavor's ``timings.csv``. No-op when writing to stdout."""
        if self._dirs is None:
            return
        for target, directory in self._dirs.items():
            with (directory / "timings.csv").open("w", newline="") as handle:
                handle.write(f"{TIMINGS_BANNER}\n")
                writer = csv.writer(handle)
                writer.writerow(["sequence", "milliseconds"])
                for header, elapsed_ms in self._timings[target]:
                    writer.writerow([header, round(elapsed_ms)])

    def __enter__(self) -> "CaidWriter":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
