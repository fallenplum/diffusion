"""CLI wrapper for the Phase1.5R2 functional repair runner."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phase1_5r2.runner import main


if __name__ == "__main__":
    main()
