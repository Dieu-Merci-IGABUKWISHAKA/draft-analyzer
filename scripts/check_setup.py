"""Fresh-setup check: Python version, packages, the GRID data folder and the caches.

Run from the project root:   python scripts/check_setup.py
Prints what is fine, what is missing, and the exact next steps. Exit code 0 when the
dashboard can run (caches that are missing are built on first use), 1 otherwise.
"""

from __future__ import annotations

import re
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MIN_PYTHON = (3, 13)
OK, MISSING, NOTE = "[ok]  ", "[MISS]", "[note]"

RUN = "python src/draft_analyzer/"
# cache file -> (what it is, the command that builds it, rough time, built automatically on first use?)
CACHES = {
    "drafts.parquet": ("drafts table", RUN + "drafts.py --rebuild", "about 40 s", True),
    "objective_events.parquet": ("early-game events", RUN + "drafts.py --rebuild", "same pass", True),
    "draft_model.joblib": ("win-probability model", RUN + "model.py", "about 1 min", True),
    "likely_eval.joblib": ("likely-pick accuracy check", RUN + "simulator.py", "about 1 min", True),
    "insights_replication.joblib": ("insight replication check", RUN + "insights.py", "~30 s", True),
    "macro_replication.joblib": ("early-game replication check", RUN + "macro.py", "seconds", True),
    "ban_backtest_cases.joblib": ("Ban plan backtest", RUN + "backtest.py --rebuild", "~10 min", False),
}


def check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) >= MIN_PYTHON
    return ok, f"Python {v.major}.{v.minor}.{v.micro}" + ("" if ok else f" (need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+)")


def requirements() -> list[tuple[str, str]]:
    """(distribution, pinned version) for each line of requirements.txt that applies here."""
    out = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        spec, _, marker = line.partition(";")
        m = re.search(r'python_version\s*<\s*"(\d+)\.(\d+)"', marker)
        if m and sys.version_info[:2] >= (int(m.group(1)), int(m.group(2))):
            continue  # e.g. zstandard is only needed before Python 3.14
        name, _, version = spec.strip().partition("==")
        out.append((name.strip(), version.strip()))
    return out


def check_packages() -> tuple[list[str], list[str]]:
    """(missing, version mismatches)."""
    missing, mismatched = [], []
    for name, pinned in requirements():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        if pinned and installed != pinned:
            mismatched.append(f"{name} {installed} (pinned {pinned})")
    return missing, mismatched


def main() -> int:
    problems, steps = 0, []
    print("Draft Tendency Analyzer: setup check\n")

    ok, text = check_python()
    print(OK if ok else MISSING, text)
    if not ok:
        problems += 1
        steps.append("Install Python 3.13 or newer, then re-run this check with it.")

    missing, mismatched = check_packages()
    if missing:
        problems += 1
        print(MISSING, "Packages not installed:", ", ".join(missing))
        steps.append("pip install -r requirements.txt")
    else:
        print(OK, f"All {len(requirements())} required packages installed")
    for m in mismatched:
        print(NOTE, "Different version than pinned:", m)

    from draft_analyzer.config import CACHE_DIR, DATA_DIR, LOL_DIR, META_DIR

    needed = [LOL_DIR / "series.csv", META_DIR / "games.csv", META_DIR / "characters.csv", META_DIR / "rosters.csv"]
    absent = [p for p in needed if not p.exists()]
    if not DATA_DIR.exists():
        problems += 1
        print(MISSING, f"GRID data folder not found: {DATA_DIR}")
        steps.append(
            "Clone the data repo next to this project (README, Setup step 1), or set STL_DATA_DIR to its folder."
        )
    elif absent:
        problems += 1
        print(MISSING, f"GRID data folder found ({DATA_DIR}) but missing: " + ", ".join(p.name for p in absent))
        steps.append("In the data repo, run: git sparse-checkout set lol meta")
    else:
        feeds = len(list(LOL_DIR.glob("*.jsonl.zst")))
        print(OK, f"GRID data folder: {DATA_DIR} ({feeds} LoL feeds)")
        if feeds == 0:
            problems += 1
            steps.append("The LoL feeds are missing: in the data repo, run git sparse-checkout set lol meta")

    if (ROOT / "stl-hackathon-2026-grid-starter").exists():
        print(MISSING, "The data repo is INSIDE this project. Move it next to the project: it must never be committed.")
        problems += 1

    print(f"\nCaches in {CACHE_DIR} (gitignored):")
    for name, (what, cmd, time, auto) in CACHES.items():
        if (CACHE_DIR / name).exists():
            print(OK, f"{what} ({name})")
        else:
            when = "built automatically on first use" if auto else "only needed to re-run the backtest"
            print(NOTE, f"{what} not built yet: {when}. To build it now: {cmd}  ({time})")
            if name == "drafts.parquet" and not absent and DATA_DIR.exists():
                steps.append("Build the drafts table: python src/draft_analyzer/drafts.py --rebuild  (about 40 s)")

    print()
    if problems:
        print(f"{problems} problem(s). Next steps:")
        for i, step in enumerate(dict.fromkeys(steps), 1):
            print(f"  {i}. {step}")
        return 1
    print("Ready. Start the dashboard with:  python -m streamlit run app/streamlit_app.py")
    if steps:
        print("Optional first:", "; ".join(dict.fromkeys(steps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
