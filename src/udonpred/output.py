"""Shared per-residue post-processing and CAID output writing.

Both runners -- the CAID precomputed-embedding one
(:mod:`udonpred.caid.predict`) and the on-the-fly one
(:mod:`udonpred.embedding.predict`) -- turn a head's raw output into the same
CAID artefacts, so that chain lives here rather than in either runner. Nothing
in this module imports ``torch``, keeping the lean CAID import boundary intact.
"""

import sys
from pathlib import Path

import numpy as np

from udonpred.fasta import format_predictions, safe_filename
from udonpred.inference import binarize_scores, normalize_scores, smooth_scores


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


class CaidWriter:
    """Write predictions in the CAID layout: ``{output}/{target}/{protein}.caid``.

    One directory per prediction head (the CAID "flavor"), holding one file per
    protein. Without an output directory everything goes to stdout instead,
    prefixed by a ``# target: <name>`` line when more than one head is running.

    Smoothing and normalization are run-wide settings, so they are fixed here
    once and applied to every :meth:`write` call.
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
