"""Project paths.

The GRID data lives outside this repo. Set STL_DATA_DIR to point at it;
otherwise it defaults to ../stl-hackathon-2026-grid-starter next to the project.
"""

import os
from pathlib import Path

# src/draft_analyzer/config.py -> project root is two levels above the package
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "stl-hackathon-2026-grid-starter"

DATA_DIR = Path(os.environ.get("STL_DATA_DIR") or _DEFAULT_DATA_DIR).expanduser().resolve()

META_DIR = DATA_DIR / "meta"
LOL_DIR = DATA_DIR / "lol"

# Derived data goes here (git-ignored)
CACHE_DIR = PROJECT_ROOT / "data_cache"

CLOUD9_LOL_TEAM = "Cloud9 Kia"
