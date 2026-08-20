"""ONNX prediction-head loading, scoring, batching, smoothing, and normalization.

This module is backbone-agnostic: it operates purely on per-residue embedding
arrays, so it is shared by both the on-the-fly runner and the CAID
precomputed-embedding runner.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d


class TargetPolicy(NamedTuple):
    """How one head's raw output maps onto the two CAID output conventions.

    Attributes:
        scale: ``(lo, hi)`` bounds of the head's raw scale, used to clamp and
            rescale onto ``[0, 1]``. ``None`` for heads that already emit a
            sigmoid probability, whose output is passed through untouched.
        flip: Whether the raw scale runs the other way (higher = more
            *ordered*), so both the score and the binary call must be inverted.
        threshold: Disorder cutoff **on the raw scale**. A residue is called
            disordered when its raw score is ``< threshold`` for a flipped head,
            and ``>= threshold`` otherwise.
    """

    scale: tuple[float, float] | None
    flip: bool
    threshold: float


# Per-head policy for the two CAID conventions: scores in [0, 1] where higher
# means more disordered, plus a binary disorder call. Thresholds are expressed
# in each head's own raw units, so they stay readable against the literature
# cutoffs and apply with or without --normalize.
TARGET_POLICIES: dict[str, TargetPolicy] = {
    # Sigmoid heads: already [0, 1] and disorder-positive.
    "trizod": TargetPolicy(None, False, 0.4),
    "trizod2": TargetPolicy(None, False, 0.4),  # same head family as trizod
    "disprot": TargetPolicy(None, False, 0.5),  # probability of a binary label
    "softdis": TargetPolicy(None, False, 0.025),
    # Unbounded regression heads.
    "chezod": TargetPolicy((-5.0, 16.15), True, 3.0),  # Z-score; higher = ordered
    "plddt": TargetPolicy((0.0, 100.0), True, 68.8),  # pLDDT; higher = ordered
    "pdbflex": TargetPolicy((0.0, 10.0), False, 2.0),  # Angstrom RMSD
    "atlas": TargetPolicy((0.0, 10.0), False, 2.0),  # Angstrom RMSF
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


def unknown_targets(targets: Iterable[str]) -> list[str]:
    """Return the requested target names that have no registered policy."""
    return [name for name in targets if name not in TARGET_POLICIES]


def target_policy(target: str) -> TargetPolicy:
    """Look up a head's policy, with a helpful error for unregistered heads."""
    try:
        return TARGET_POLICIES[target]
    except KeyError:
        raise ValueError(
            f"No policy registered for target {target!r}. Known targets: "
            f"{', '.join(sorted(TARGET_POLICIES))}."
        ) from None


def normalize_scores(scores: np.ndarray, target: str) -> np.ndarray:
    """Map raw head output onto the CAID convention: [0, 1], higher = disorder.

    Heads that already emit a sigmoid probability (see ``TARGET_POLICIES``) are
    returned unchanged. The rest are clamped to their fixed a-priori scale,
    rescaled onto [0, 1], and inverted if their scale runs the other way
    (``chezod``, ``plddt``). The input shape is preserved.
    """
    policy = target_policy(target)
    if policy.scale is None:
        return scores
    lo, hi = policy.scale
    out = (np.clip(np.asarray(scores, dtype=np.float64), lo, hi) - lo) / (hi - lo)
    return 1.0 - out if policy.flip else out


def binarize_scores(scores: np.ndarray, target: str) -> np.ndarray:
    """Call each residue disordered (1) or ordered (0) from **raw** scores.

    The threshold lives on the head's raw scale, so this must be applied to the
    scores as the head emits them — before :func:`normalize_scores`. Flipped
    heads (``chezod``, ``plddt``) are order-positive, so for them disorder is
    *below* the cutoff. The input shape is preserved.
    """
    policy = target_policy(target)
    raw = np.asarray(scores, dtype=np.float64)
    calls = raw < policy.threshold if policy.flip else raw >= policy.threshold
    return calls.astype(np.int8)


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
