"""Guard the lean, offline boundary of the shipped CAID inference path.

Importing :mod:`udonpred.caid.predict` (and the core it depends on, including
the head resolver :mod:`udonpred.heads`) must never pull in ``torch`` or
``huggingface_hub``. The CAID4 container installs only the lean dependency set
and consumes heads baked in at build time, so a stray top-level import of either
would break it. Run in a subprocess so the check is independent of what the test
process has already imported.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"


def test_caid_inference_path_never_imports_torch():
    code = (
        "import udonpred.caid.predict\n"
        "import udonpred.caid.embeddings\n"
        "import udonpred.fasta\n"
        "import udonpred.inference\n"
        "import udonpred.heads\n"
        "import udonpred.datasets\n"
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch leaked into the CAID path'\n"
        "assert 'huggingface_hub' not in sys.modules, "
        "'huggingface_hub leaked into the CAID path'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr


def test_shared_output_and_cli_modules_stay_torch_free():
    """The shared modules are on the CAID path, so they must not pull in torch."""
    code = (
        "import udonpred.output\n"
        "import udonpred.cli\n"
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch leaked into the shared modules'\n"
        "assert 'huggingface_hub' not in sys.modules, "
        "'huggingface_hub leaked into the shared modules'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
