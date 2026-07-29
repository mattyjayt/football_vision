"""Pytest configuration for the football_vision test suite.

`footlab` is an installed package (pyproject: packages = ["src/footlab"]),
but `detection_pipeline` lives at the repo root and is imported the same way
the scripts do it — via the repo root on sys.path. Mirror that here so tests
can import both packages identically to how the demos use them.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
