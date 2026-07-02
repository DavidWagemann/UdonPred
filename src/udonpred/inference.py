"""ONNX prediction-head loading, scoring, batching, and score smoothing.

This module is backbone-agnostic: it operates purely on per-residue embedding
arrays, so it is shared by both the on-the-fly runner and the CAID
precomputed-embedding runner.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d


def load_head(
    onnx_path: Path,
    device: str,
    threads: Optional[int] = None,
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


def iter_batches(items: List[Tuple[str, str]], max_total_len: int):
    """Yield batches of ``(header, sequence)`` capped by total residue count."""
    batch: List[Tuple[str, str]] = []
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


def count_batches(items: List[Tuple[str, str]], max_total_len: int) -> int:
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
