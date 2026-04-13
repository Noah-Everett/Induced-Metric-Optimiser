"""Pytest configuration — add repo paths so imports work."""

import sys
import os

_REPO = os.path.join(os.path.dirname(__file__), "..")
for subdir in ["", "parameters", "analysis"]:
    p = os.path.join(_REPO, subdir)
    if p not in sys.path:
        sys.path.insert(0, p)
