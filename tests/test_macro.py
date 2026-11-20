"""Tests for early-game macro extraction (drafts.scan_series) and draft_analyzer.macro.
Synthetic data only: a tiny made-up feed and made-up objective events."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from draft_analyzer import insights as I
from draft_analyzer import macro as MA
from draft_analyzer.drafts import scan_series

zstandard = pytest.importorskip("zstandard")


# --- extraction from a feed ---------------------------------------------------------------
def _event(etype, actor, target, game="g1", seconds=0):
    return {
        "type": etype,
        "actor": actor,
        "target": target,
        "seriesStateDelta": {"games": [{"id": game}]},
        "seriesState": {"games": [{"id": game, "clock": {"currentSeconds": seconds}}]},
    }


def _player(pid, team):
    return {"type": "player", "id": pid, "state": {"teamId": team}}


def test_scan_series_reads_objectives(tmp_path):
    envs = [
        {
            "occurredAt": "t1",
            "events": [_event("player-killed-player", _player("p1", "A"), _player("p9", "B"), seconds=200)],
        },
        {
            "occurredAt": "t2",
            "events": [_event("player-multikilled-player", _player("p1", "A"), _player("p9", "B"), seconds=201)],
        },
        {
            "occurredAt": "t3",
            "events": [
                _event(
                    "player-completed-slayMountainDrake", _player("p2", "B"), {"id": "slayMountainDrake"}, seconds=400
                )
            ],
        },
        {
            "occurredAt": "t4",
            "events": [
                _event(
                    "player-completed-slayRiftScuttlerTop",
                    _player("p2", "B"),
                    {"id": "slayRiftScuttlerTop"},
                    seconds=210,
                )
            ],
        },
        {
            "occurredAt": "t5",
            "events": [
                _event("team-destroyed-tower", {"type": "team", "id": "A"}, {"id": "red-turret-bot-1"}, seconds=800)
            ],
        },
        {
            "occurredAt": "t6",
            "events": [
                _event("player-!completed-slayVoidGrub", _player("p3", "A"), {"id": "slayVoidGrub"}, seconds=500)
            ],
        },
    ]
    path = tmp_path / "feed.jsonl.zst"
    raw = "\n".join(json.dumps(e) for e in envs).encode()
    path.write_bytes(zstandard.ZstdCompressor().compress(raw))
    state, events = scan_series(path)
    types = [e["type"] for e in events]
    assert types == [
        "player-killed-player",
        "player-completed-slayMountainDrake",
        "team-destroyed-tower",
        "player-completed-slayVoidGrub",
    ]  # no multikill, no scuttler
    assert [e["team_id"] for e in events] == ["A", "B", "A", "A"]
    assert [e["seconds"] for e in events] == [200, 400, 800, 500]
    assert [e["undo"] for e in events] == [False, False, False, True]
    assert state is not None


# --- firsts per game ----------------------------------------------------------------------
GAMES = pd.DataFrame({"blue_team": ["A", "A"], "red_team": ["B", "B"]}, index=["g1", "g2"])


def _ev(rows):
    df = pd.DataFrame(rows, columns=["game_id", "type", "team_id", "target_id", "seconds", "undo"])
    return df.assign(order=range(len(df)))


def test_first_objectives():
    ev = _ev(
        [
            ("g1", "player-completed-slayDragon", "B", "slayDragon", 420, False),
            ("g1", "player-completed-slayInfernalDrake", "B", "x", 420, False),
            ("g1", "player-completed-slayDragon", "A", "slayDragon", 900, False),
            ("g1", "player-killed-player", "Z", None, 100, False),  # not a team in this game
            ("g1", "player-killed-player", "A", None, 150, False),
            ("g1", "player-destroyed-tower", "B", "blue-turret-top-1", 700, False),
            ("g1", "player-destroyed-tower", "B", "red-turret-mid-1", 650, False),  # credited to the wrong side
            ("g2", "player-completed-slayMountainDrake", "A", "x", 380, False),  # drake-type only (early 2025)
        ]
    )
    f = MA.first_objectives(ev, GAMES)
    assert f.loc["g1", "first_dragon_team"] == "B" and f.loc["g1", "first_dragon_s"] == 420
    assert f.loc["g1", "first_blood_team"] == "A" and f.loc["g1", "first_blood_s"] == 150
    assert f.loc["g1", "first_tower_team"] == "A"  # red tower fell first -> blue (A) took it
    assert f.loc["g2", "first_dragon_team"] == "A"
    assert np.isnan(f.loc["g2", "first_herald_s"]) and pd.isna(f.loc["g2", "first_herald_team"])


def test_undo_removes_the_matching_event():
    ev = _ev(
        [
            ("g1", "player-completed-slayVoidGrub", "A", "slayVoidGrub", 500, False),
            ("g1", "player-completed-slayVoidGrub", "A", "slayVoidGrub", 500, True),
            ("g1", "player-completed-slayVoidGrub", "B", "slayVoidGrub", 560, False),
        ]
    )
    assert MA.first_objectives(ev, GAMES).loc["g1", "first_grubs_team"] == "B"


def test_mmss_and_holds_up():
    assert MA.mmss(65) == "1:05" and MA.mmss(float("nan")) == "–"
    assert I.holds_up(0.5, 0.01) and not I.holds_up(0.5, -0.01) and not I.holds_up(0.2, 0.1)


# --- team profiles on the synthetic drafts ------------------------------------------------
@pytest.fixture(scope="module")
def macro(synthetic_drafts):
    """Planted habits: blue always takes first dragon; a team takes first grubs exactly
    when its jungler is C8; red always takes first blood at 3:00."""
    scout = I.Scout(synthetic_drafts)
    g, a = scout.games, scout.actions
    jg = a[(a["action"] == "pick") & (a["role"] == "jungle")].set_index(["game_id", "team_id"])["champion_id"]
    rows = []
    for gid, r in g.iterrows():
        rows.append((gid, "player-completed-slayDragon", r["blue_team"], "slayDragon", 400, False))
        rows.append((gid, "player-killed-player", r["red_team"], None, 180, False))
        grub = r["blue_team"] if jg.get((gid, r["blue_team"])) == "C8" else r["red_team"]
        rows.append((gid, "player-completed-slayVoidGrub", grub, "slayVoidGrub", 500, False))
    return MA.Macro(scout, _ev(rows))


def test_profile(macro):
    tg = macro.scout.team_games("T0", None)
    p = macro.profile("T0", tg).set_index("key")
    blue_share = (tg["side"] == "blue").mean()
    assert p.loc["first_dragon", "rate"] == pytest.approx(blue_share)
    assert p.loc["first_blood", "rate"] == pytest.approx(1 - blue_share)
    assert p.loc["first_blood", "their_median_s"] == 180
    assert p.loc["first_dragon", "games"] == len(tg)
    assert p.loc["first_herald", "games"] == 0  # never happened
    lo, hi = I.wilson(int(p.loc["first_dragon", "took"]), len(tg))
    assert (p.loc["first_dragon", "lo"], p.loc["first_dragon", "hi"]) == pytest.approx((lo, hi))


def test_jungler_link_finds_planted_habit(macro):
    tg = macro.scout.team_games("T0", None)
    links = macro.jungler_links("T0", tg)
    c8 = links[(links["jungler_id"] == "C8") & (links["key"] == "first_grubs")]
    if c8.empty:
        pytest.skip("C8 not played often enough by T0 in the synthetic data")
    assert c8["diff"].iloc[0] > 0.3 and c8["confidence"].iloc[0] > 0.9


def test_replication_and_summary(macro):
    rep = MA.replication(macro, splits=("2025-03-01",))
    assert {"kind", "key", "early_effect", "late_effect"} <= set(rep.columns)
    summary = MA.replication_summary(rep)
    assert summary["survives"].between(0, 1).all()
    s = macro.summary("T0", None, replication=summary)
    assert {"show_rate", "show_timing"} <= set(s["profile"].columns)
    shown = s["profile"][s["profile"]["show_rate"]]
    assert (shown["rate_conf"] >= I.MIN_CONFIDENCE).all()
    assert all(line.endswith(".") and "games" in line for line in s["jungler_lines"])
