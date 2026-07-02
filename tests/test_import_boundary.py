"""Guard the torch-free boundary of the shipped CAID inference path.

Importing :mod:`udonpred.caid.predict` (and the core it depends on) must never
pull in ``torch``. The CAID4 container installs only the lean dependency set, so
a stray top-level ``import torch`` anywhere on this path would break it. Run in
a subprocess so the check is independent of what the test process has imported.
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
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch leaked into the CAID path'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, result.stderr
