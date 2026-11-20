"""One row per pick/ban for every LoL game in the GRID data.

Source of truth: the final ``seriesState.games[].draftActions`` of each feed.
It already has "!" undo events and rolled-back games resolved, and in a few
games it holds bans that the ``team-*-character`` event stream missed.
Patch, sides, team names and the winner come from ``meta/games.csv``;
champion names from ``meta/characters.csv``. Everything joins on IDs.
Picks also get the player and an inferred role from ``roles.py``.

Usage:
    python src/draft_analyzer/drafts.py            # load cache (build if missing)
    python src/draft_analyzer/drafts.py --rebuild  # re-stream every feed
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
from collections import deque
from pathlib import Path
from typing import TextIO

import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer.config import CACHE_DIR, DATA_DIR, LOL_DIR, META_DIR  # noqa: E402
from draft_analyzer.roles import infer_roles  # noqa: E402

DRAFTS_PATH = CACHE_DIR / "drafts.parquet"
OBJECTIVES_PATH = CACHE_DIR / "objective_events.parquet"  # early-game events, see macro.py

# Event types collected for macro.py. Lines are pre-filtered on substrings of the event
# type ("slayDragon" alone also appears inside every later snapshot's objective lists),
# then events are kept by full type. Early-2025 feeds have only the drake-type events
# (slayMountainDrake, ...), later ones also the generic slayDragon, so both are kept.
OBJECTIVE_MARKERS = (
    "-completed-slay",
    "-!completed-slay",
    "destroyed-tower",
    "player-killed-player",
    "player-!killed-player",
)  # "!" = undo of an earlier event
OBJECTIVE_TYPE_RE = re.compile(
    r"completed-slay(Dragon|\w+Drake|VoidGrub|RiftHerald)$|destroyed-tower$"
    r"|^player-killed-player$"
)

COLUMNS = [
    "series_id",
    "game_id",
    "game_number",
    "event",
    "stage",
    "date",
    "patch",
    "team_id",
    "team_name",
    "opponent_id",
    "opponent_name",
    "side",
    "action",
    "draft_order",
    "champion_id",
    "champion_name",
    "game_winner_team_id",
]
# Picks only (see roles.py); empty for bans.
ROLE_COLUMNS = ["player_id", "player_name", "role_inferred", "role_source"]
COLUMNS += ROLE_COLUMNS

# Standard pro tournament draft: slot -> (drafter, action), where "F" is the
# team with first pick and "S" the other. Until 2025 F was always blue side;
# from 2026 ("first selection") F is often red, so slots are keyed on F/S.
DRAFT_TEMPLATE: list[tuple[str, str]] = [
    ("F", "ban"),
    ("S", "ban"),
    ("F", "ban"),
    ("S", "ban"),
    ("F", "ban"),
    ("S", "ban"),
    ("F", "pick"),
    ("S", "pick"),
    ("S", "pick"),
    ("F", "pick"),
    ("F", "pick"),
    ("S", "pick"),
    ("S", "ban"),
    ("F", "ban"),
    ("S", "ban"),
    ("F", "ban"),
    ("S", "pick"),
    ("F", "pick"),
    ("F", "pick"),
    ("S", "pick"),
]

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


def scan_series(path: Path, objectives: bool = True) -> tuple[dict | None, list[dict]]:
    """One pass over a feed: the final seriesState (as final_series_state) and, if
    `objectives`, every OBJECTIVE_TYPES event with its game, team and game-clock seconds.
    Undo events ("!") are kept with undo=True; macro.py applies them."""
    tail: deque[str] = deque(maxlen=20)
    events: list[dict] = []
    with open_zst(path) as f:
        for line in f:
            if '"seriesState"' in line:
                tail.append(line)
            if objectives and any(m in line for m in OBJECTIVE_MARKERS):
                env = json.loads(line)
                for e in env.get("events", []):
                    if not OBJECTIVE_TYPE_RE.search(e["type"].replace("!", "")):
                        continue
                    gids = [g["id"] for g in (e.get("seriesStateDelta") or {}).get("games", [])]
                    if not gids:
                        continue
                    clocks = {
                        g["id"]: (g.get("clock") or {}).get("currentSeconds")
                        for g in (e.get("seriesState") or {}).get("games", [])
                    }
                    actor = e.get("actor") or {}
                    team = (
                        (actor.get("state") or {}).get("teamId") if actor.get("type") == "player" else actor.get("id")
                    )
                    events.append(
                        {
                            "game_id": gids[0],
                            "type": e["type"].replace("!", ""),
                            "undo": "!" in e["type"],
                            "actor_kind": actor.get("type"),
                            "team_id": team,
                            "target_id": (e.get("target") or {}).get("id"),
                            "seconds": clocks.get(gids[0]),
                            "occurred_at": env.get("occurredAt"),
                            "order": len(events),
                        }
                    )
    for line in reversed(tail):
        for event in reversed(json.loads(line).get("events", [])):
            if event.get("seriesState"):
                return event["seriesState"], events
    return None, events


def final_series_state(path: Path) -> dict | None:
    """Return the seriesState of the last event that has one.

    Only the tail of the file is parsed as JSON, which is far faster than
    decoding every envelope.
    """
    tail: deque[str] = deque(maxlen=20)
    with open_zst(path) as f:
        for line in f:
            if '"seriesState"' in line:
                tail.append(line)
    for line in reversed(tail):
        for event in reversed(json.loads(line).get("events", [])):
            if event.get("seriesState"):
                return event["seriesState"]
    return None


def assign_draft_slots(actions: list[tuple[str, str]]) -> list[int] | None:
    """Map (team_id, action) pairs, in draft order, to standard slots 1-20.

    The first team to pick is "F". Skipped bans leave a gap rather than
    shifting later slots. Returns None if the sequence doesn't fit.
    """
    first_picker = next((team for team, act in actions if act == "pick"), None)
    slots, pos = [], 0
    for team, action in actions:
        act = ("F" if team == first_picker else "S", action)
        while pos < len(DRAFT_TEMPLATE) and DRAFT_TEMPLATE[pos] != act:
            pos += 1
        if pos == len(DRAFT_TEMPLATE):
            return _slots_by_occurrence(actions, first_picker)
        slots.append(pos + 1)
        pos += 1
    return slots


def _slots_by_occurrence(actions: list[tuple[str, str]], first_picker: str | None) -> list[int] | None:
    """Fallback for feeds that logged actions slightly out of order: a team's
    k-th ban (or pick) goes to its k-th ban (or pick) slot."""
    free: dict[tuple[str, str], list[int]] = {}
    for slot, key in enumerate(DRAFT_TEMPLATE, 1):
        free.setdefault(key, []).append(slot)
    slots = []
    for team, action in actions:
        queue = free.get(("F" if team == first_picker else "S", action))
        if not queue:
            return None
        slots.append(queue.pop(0))
    return slots


def _load_meta() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    games = pd.read_csv(META_DIR / "games.csv", dtype=str, keep_default_na=False)
    games = games[games["title"] == "lol"].set_index("game_id")
    series = pd.read_csv(LOL_DIR / "series.csv", dtype=str, keep_default_na=False)
    chars = pd.read_csv(META_DIR / "characters.csv", dtype=str, keep_default_na=False)
    champ_names = dict(
        zip(
            chars.loc[chars["title"] == "lol", "character_id"], chars.loc[chars["title"] == "lol", "name"], strict=False
        )
    )
    return games, series, champ_names


def build_drafts(finished_only: bool = True, objectives_out: list | None = None) -> pd.DataFrame:
    """Stream every series in lol/series.csv and return one row per draft action. If
    `objectives_out` is a list, the same pass appends every series' objective events to it."""
    games, series, champ_names = _load_meta()
    rows: list[dict] = []
    seen_games: set[str] = set()
    warnings: list[str] = []
    t0 = time.time()
    n = len(series)

    for i, s in enumerate(series.itertuples(index=False), 1):
        print(f"\r[{i}/{n}] series {s.series_id}  ({time.time() - t0:.0f}s)", end="", file=sys.stderr, flush=True)
        state, obj = scan_series(DATA_DIR / s.file, objectives=objectives_out is not None)
        if objectives_out is not None:
            objectives_out.extend({**e, "series_id": s.series_id} for e in obj)
        if state is None:
            warnings.append(f"series {s.series_id}: no seriesState in feed")
            continue

        for g in state.get("games", []):
            gid = g["id"]
            if gid not in games.index:
                warnings.append(f"series {s.series_id} game {gid}: in feed but not in meta/games.csv, skipped")
                continue
            meta = games.loc[gid]
            if finished_only and meta["finished"] != "True":
                warnings.append(f"series {s.series_id} game {meta['game_number']}: not finished in meta, skipped")
                continue
            seen_games.add(gid)

            teams = {
                meta["team1_id"]: (meta["team1_name"], meta["team1_side"]),
                meta["team2_id"]: (meta["team2_name"], meta["team2_side"]),
            }
            actions = sorted(g.get("draftActions") or [], key=lambda a: int(a["sequenceNumber"]))
            if len(actions) != 20:
                warnings.append(f"series {s.series_id} game {meta['game_number']}: {len(actions)} draft actions")
            slots = assign_draft_slots([(a["drafter"]["id"], a["type"]) for a in actions])
            if slots is None:
                warnings.append(
                    f"series {s.series_id} game {meta['game_number']}: draft doesn't fit standard order, "
                    "using sequenceNumber"
                )
                slots = [int(a["sequenceNumber"]) for a in actions]

            date = (meta["started_at"] or s.scheduled_start)[:10]
            for a, slot in zip(actions, slots, strict=False):
                team_id = a["drafter"]["id"]
                opp_id = next((t for t in teams if t != team_id), "")
                champ_id = a["draftable"]["id"]
                rows.append(
                    {
                        "series_id": s.series_id,
                        "game_id": gid,
                        "game_number": int(meta["game_number"]),
                        "event": s.event,
                        "stage": s.stage,
                        "date": date,
                        "patch": meta["patch"] or None,
                        "team_id": team_id,
                        "team_name": teams.get(team_id, ("", ""))[0],
                        "opponent_id": opp_id,
                        "opponent_name": teams.get(opp_id, ("", ""))[0],
                        "side": teams.get(team_id, ("", ""))[1],
                        "action": a["type"],
                        "draft_order": slot,
                        "champion_id": champ_id,
                        # characters.csv only lists champions that were played; a
                        # ban-only champion falls back to the feed's name for the ID
                        "champion_name": champ_names.get(champ_id) or a["draftable"].get("name"),
                        "game_winner_team_id": meta["winner_team_id"] or None,
                    }
                )
    print(file=sys.stderr)

    missing = games.index.difference(list(seen_games))
    for gid in missing:
        m = games.loc[gid]
        if finished_only and m["finished"] != "True":
            continue
        warnings.append(f"series {m['series_id']} game {m['game_number']}: in meta/games.csv but no draft in feed")
    for w in warnings:
        print("  warning:", w, file=sys.stderr)

    df = pd.DataFrame(rows, columns=COLUMNS[: -len(ROLE_COLUMNS)])

    # Picks get the player who played the champion and their inferred role
    # (see roles.py: there is no role field in the data). Bans stay empty.
    roles = infer_roles().rename(columns={"character_id": "champion_id"})
    df = df.merge(roles, on=["game_id", "team_id", "champion_id"], how="left")
    df.loc[df["action"] == "ban", ROLE_COLUMNS] = None
    unmatched = df[(df["action"] == "pick") & df["role_inferred"].isna()]
    if len(unmatched):
        print(f"  warning: {len(unmatched)} picks have no roster match for a role", file=sys.stderr)

    df["date"] = pd.to_datetime(df["date"]).dt.date
    for col in (
        "event",
        "stage",
        "patch",
        "side",
        "action",
        "team_name",
        "opponent_name",
        "champion_name",
        "role_inferred",
        "role_source",
    ):
        df[col] = df[col].astype("category")
    return df.sort_values(["date", "series_id", "game_number", "draft_order"]).reset_index(drop=True)


def load_drafts(rebuild: bool = False) -> pd.DataFrame:
    """Load the drafts table from the cache, building it if needed (or if rebuild=True)."""
    if rebuild or not DRAFTS_PATH.exists():
        objectives: list[dict] = []
        df = build_drafts(objectives_out=objectives)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(DRAFTS_PATH, index=False)
        pd.DataFrame(objectives).to_parquet(OBJECTIVES_PATH, index=False)
        print(
            f"Saved {len(df)} rows to {DRAFTS_PATH} and {len(objectives)} objective events to {OBJECTIVES_PATH}",
            file=sys.stderr,
        )
        return df
    return pd.read_parquet(DRAFTS_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rebuild", action="store_true", help="re-stream every feed instead of loading the cache")
    args = parser.parse_args()
    df = load_drafts(rebuild=args.rebuild)
    print(f"{len(df)} draft actions, {df['game_id'].nunique()} games, {df['series_id'].nunique()} series")


if __name__ == "__main__":
    main()
