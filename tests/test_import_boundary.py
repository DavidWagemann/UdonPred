"""Guard the lean, torch-free boundary of the inference core.

``pip install udonpred-comp`` installs only the lean dependency set, so importing
the core modules must never pull in ``torch`` or ``huggingface_hub`` (the latter
only when a Hub download actually happens). Run in a subprocess so the check is
independent of what the test process has already imported.
"""

import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

CORE_MODULES = (
    "udonpred.cli",
    "udonpred.datasets",
    "udonpred.fasta",
    "udonpred.filtering",
    "udonpred.heads",
    "udonpred.inference",
    "udonpred.output",
)


def test_core_never_imports_torch_or_hub():
    code = "".join(f"import {name}\n" for name in CORE_MODULES) + (
        "import sys\n"
        "assert 'torch' not in sys.modules, 'torch leaked into the core'\n"
        "assert 'huggingface_hub' not in sys.modules, "
        "'huggingface_hub leaked into the core'\n"
    )
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
