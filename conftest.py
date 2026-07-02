"""Shared test fixtures."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def weights_dir():
    """A directory containing the ONNX prediction heads.

    Prefers a local ``weights/`` checkout; otherwise resolves the pinned release
    from the Hub (cached). Skips the test if neither is available (e.g. offline
    with no local heads), so the suite degrades gracefully.
    """
    local = ROOT / "weights"
    if local.is_dir() and any(local.glob("*.onnx")):
        return local

    try:
        from udonpred.heads import resolve_model_dir

        resolved = resolve_model_dir(None)
        if any(Path(resolved).glob("*.onnx")):
            return Path(resolved)
    except Exception:  # network/hub unavailable
        pass

    pytest.skip("No ONNX prediction heads available locally or from the Hub")
