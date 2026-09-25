"""Shared helper for the command-line entry points in this directory.

Each entry point forwards its arguments to an implementation module in
`tools/_impl/`, which is where the actual code lives.  Keeping the implementation in a
private subpackage means the repository exposes a handful of clearly named tools instead
of three dozen scripts.
"""
from __future__ import annotations

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def run(module: str, argv=None) -> None:
    """Import `tools._impl.<module>` and call its main() with the given arguments."""
    args = list(sys.argv[1:] if argv is None else argv)
    sys.argv = [module] + args
    importlib.import_module(f"tools._impl.{module}").main()
