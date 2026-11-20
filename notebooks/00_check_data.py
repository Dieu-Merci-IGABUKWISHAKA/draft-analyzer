"""Smoke test: confirm the GRID data is reachable and readable.

Loads meta tables, then streams one Cloud9 Kia match feed (never written
to disk decompressed) and summarises its events.
"""

import io
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TextIO

import pandas as pd

# Make src/ importable when run as a plain script
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draft_analyzer.config import CLOUD9_LOL_TEAM, DATA_DIR, LOL_DIR, META_DIR  # noqa: E402

try:  # Python 3.14+
    from compression import zstd as _zstd

    def open_zst(path: Path) -> TextIO:
        return _zstd.open(path, "rt", encoding="utf-8")
except ImportError:
    import zstandard

    def open_zst(path: Path) -> TextIO:
        return io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(open(path, "rb"), closefd=True),
            encoding="utf-8",
        )


def name_of(entity: dict) -> str:
    state = entity.get("state") or {}
    return state.get("name") or entity.get("name") or entity.get("id", "?")


print(f"Data dir: {DATA_DIR}")

# --- meta tables -----------------------------------------------------------
games = pd.read_csv(META_DIR / "games.csv", dtype=str)
games = games[games["title"] == "lol"]
rosters = pd.read_csv(META_DIR / "rosters.csv", dtype=str)
rosters = rosters[rosters["title"] == "lol"]
series = pd.read_csv(LOL_DIR / "series.csv", dtype=str)

print(f"\nLoL games: {len(games)}  (series: {games['series_id'].nunique()}, roster rows: {len(rosters)})")

games = games.merge(series[["series_id", "event", "file"]], on="series_id", how="left")
print("\nEvents:")
for event, n in games["event"].value_counts().sort_index().items():
    print(f"  {event}: {n} games")

# --- Cloud9 Kia ------------------------------------------------------------
c9 = games[(games["team1_name"] == CLOUD9_LOL_TEAM) | (games["team2_name"] == CLOUD9_LOL_TEAM)]
print(f"\n{CLOUD9_LOL_TEAM} games: {len(c9)} across {c9['series_id'].nunique()} series")

# --- stream one match feed -------------------------------------------------
row = c9.sort_values("started_at").iloc[0]
feed = DATA_DIR / row["file"]
print(f"\nStreaming series {row['series_id']} ({row['event']}): {feed.name}")

type_counts: Counter[str] = Counter()
draft_examples: list[str] = []
n_envelopes = 0
with open_zst(feed) as f:
    for line in f:
        n_envelopes += 1
        for ev in json.loads(line).get("events", []):
            etype = ev["type"]
            type_counts[etype] += 1
            if etype in ("team-picked-character", "team-banned-character") and len(draft_examples) < 8:
                draft_examples.append(f"  {etype:24s} {name_of(ev['actor']):12s} -> {name_of(ev['target'])}")

print(f"Envelopes: {n_envelopes}, events: {sum(type_counts.values())}, distinct types: {len(type_counts)}")
print("\nEvent types:")
for etype, n in type_counts.most_common():
    print(f"  {n:7d}  {etype}")

print("\nSample draft events:")
print("\n".join(draft_examples) if draft_examples else "  (none found)")
