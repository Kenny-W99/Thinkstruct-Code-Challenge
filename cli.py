#!/usr/bin/env python
"""`python cli.py ...` works from the repo root even without `pip install -e .`"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from patent_search.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
