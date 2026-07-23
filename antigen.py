"""Module entry point for running the Antigen command-line interface.

This thin shim keeps ``python -m antigen`` on the same implementation path as
the installed console script, so their arguments and output stay consistent.
"""
from __future__ import annotations

from cli import main

if __name__ == "__main__":
    raise SystemExit(main())
