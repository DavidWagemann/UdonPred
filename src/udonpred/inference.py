"""ONNX prediction-head loading, scoring, batching, smoothing, and normalization.

This module is backbone-agnostic: it operates purely on per-residue embedding
arrays, so it is shared by both the on-the-fly runner and the CAID
precomputed-embedding runner.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d

# Per-head policy for mapping raw head output onto the CAID convention: scores
# in [0, 1] where higher means more disordered. ``None`` means the head already
# emits a sigmoid probability with disorder-positive polarity, so its output is
# passed through untouched. Otherwise ``(lo, hi, flip)`` clamps to ``[lo, hi]``,
# rescales onto [0, 1], and inverts if ``flip`` (i.e. the head's raw scale runs
# the other way, higher = more ordered).
TARGET_NORMALIZATION: dict[str, tuple[float, float, bool] | None] = {
    "trizod": None,
    "trizod2": None,  # same sigmoid head family as trizod
    "disprot": None,
    "softdis": None,
    "chezod": (-5.0, 16.15, True),  # CheZOD Z-score; higher = more ordered
    "plddt": (0.0, 100.0, True),  # pLDDT; higher = more confident/ordered
    "pdbflex": (0.0, 10.0, False),  # Angstrom RMSD
    "atlas": (0.0, 10.0, False),  # Angstrom RMSF
}


def load_head(
    onnx_path: Path,
    device: str,
    threads: int | None = None,
) -> ort.InferenceSession:
    """Load an ONNX prediction head.

    Args:
        onnx_path: Path to the ``{target}.onnx`` head.
        device: ``"cpu"`` or ``"cuda"`` (selects the execution provider).
        threads: Optional cap on intra/inter-op threads (CPU). ``None`` lets
            onnxruntime decide.
    """
    onnx_path = Path(onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX head not found: {onnx_path}")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if device == "cuda"
        else ["CPUExecutionProvider"]
    )
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    if threads and threads > 0:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
    return ort.InferenceSession(
        str(onnx_path), sess_options=opts, providers=providers
    )


def score_embeddings(
    head: ort.InferenceSession,
    embeddings: np.ndarray,
) -> np.ndarray:
    """Run the ONNX head on a ``(batch, seq_len, input_dim)`` embedding array.

    The embedding array is cast to ``float32`` (the dtype the heads were
    exported with). Returns the raw head output, typically shaped
    ``(batch, seq_len, 1)``.
    """
    input_name = head.get_inputs()[0].name
    output_name = head.get_outputs()[0].name
    emb = np.ascontiguousarray(embeddings, dtype=np.float32)
    return head.run([output_name], {input_name: emb})[0]


def smooth_scores(scores: np.ndarray, sigma: float) -> np.ndarray:
    """Apply 1-D Gaussian smoothing along the sequence axis.

    ``sigma <= 0`` disables smoothing and returns the input unchanged.
    """
    if sigma is None or sigma <= 0:
        return scores
    return gaussian_filter1d(
        np.asarray(scores, dtype=np.float64), sigma=sigma, axis=0
    )


def unknown_normalization_targets(targets: Iterable[str]) -> list[str]:
    """Return the requested target names that have no normalization policy."""
    return [name for name in targets if name not in TARGET_NORMALIZATION]


def normalize_scores(scores: np.ndarray, target: str) -> np.ndarray:
    """Map raw head output onto the CAID convention: [0, 1], higher = disorder.

    Heads that already emit a sigmoid probability (see ``TARGET_NORMALIZATION``)
    are returned unchanged. The rest are clamped to their fixed a-priori scale,
    rescaled onto [0, 1], and inverted if their scale runs the other way
    (``chezod``, ``plddt``). The input shape is preserved.
    """
    if target not in TARGET_NORMALIZATION:
        raise ValueError(
            f"No normalization policy for target {target!r}. Known targets: "
            f"{', '.join(sorted(TARGET_NORMALIZATION))}."
        )
    policy = TARGET_NORMALIZATION[target]
    if policy is None:
        return scores
    lo, hi, flip = policy
    out = (np.clip(np.asarray(scores, dtype=np.float64), lo, hi) - lo) / (hi - lo)
    return 1.0 - out if flip else out


def iter_batches(items: list[tuple[str, str]], max_total_len: int):
    """Yield batches of ``(header, sequence)`` capped by total residue count."""
    batch: list[tuple[str, str]] = []
    total_len = 0
    for header, seq in items:
        seq_len = len(seq)
        if batch and total_len + seq_len > max_total_len:
            yield batch
            batch = []
            total_len = 0
        batch.append((header, seq))
        total_len += seq_len
        if total_len >= max_total_len:
            yield batch
            batch = []
            total_len = 0
    if batch:
        yield batch


def count_batches(items: list[tuple[str, str]], max_total_len: int) -> int:
    """Count how many batches ``iter_batches`` will yield (for progress bars)."""
    count = 0
    total_len = 0
    for _, seq in items:
        seq_len = len(seq)
        if count == 0 and total_len == 0 and seq_len == 0:
            continue
        if total_len and total_len + seq_len > max_total_len:
            count += 1
            total_len = 0
        total_len += seq_len
        if total_len >= max_total_len:
            count += 1
            total_len = 0
    if total_len:
        count += 1
    return count
