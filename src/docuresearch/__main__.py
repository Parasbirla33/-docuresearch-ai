"""Enables `python -m docuresearch`."""

from __future__ import annotations

import sys

from docuresearch.cli import main

if __name__ == "__main__":
    sys.exit(main())
