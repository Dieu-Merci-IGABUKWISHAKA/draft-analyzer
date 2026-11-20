"""Tests for draft_analyzer.insights (statistics, cards, replication), on synthetic data."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from draft_analyzer import insights as I


@pytest.fixture(scope="module")
def scout(synthetic_drafts):
    return I.Scout(synthetic_drafts)


# --- statistics ---------------------------------------------------------------------------
def test_wilson_interval():
    lo, hi = I.wilson(7, 10)
    assert lo < 0.7 < hi and 0 <= lo and hi <= 1
    assert I.wilson(0, 10)[0] == 0 and I.wilson(10, 10)[1] == 1
    assert math.isnan(I.wilson(0, 0)[0])
    lo_s, hi_s = I.wilson(70, 100)  # more games, narrower interval
    assert hi_s - lo_s < hi - lo
    # Known value: 5 of 10 at 90% -> about 0.27-0.73
    assert I.wilson(5, 10) == pytest.approx((0.2693, 0.7307), abs=1e-3)


def test_two_prop_and_one_prop():
    diff, conf = I.two_prop(5, 10, 5, 10)
    assert diff == 0 and conf == pytest.approx(0)
    diff, conf = I.two_prop(40, 50, 10, 50)
    assert diff == pytest.approx(0.6) and conf > 0.999
    assert I.two_prop(1, 0, 1, 2) == (pytest.approx(float("nan"), nan_ok=True), 0.0)
    assert I.one_prop(6, 10, 0.6) == pytest.approx(0)
    assert I.one_prop(10, 10, 0.6) > 0.9


def test_entropy():
    assert I.entropy_bits([5]) == 0
    assert I.entropy_bits([1000, 1000, 1000, 1000]) == pytest.approx(2, abs=0.01)
    assert I.entropy_bits([10, 1]) < I.entropy_bits([10, 10])
    assert I.entropy_bits([1] * 30) == pytest.approx(math.log2(30))  # capped at what was seen


# --- cards --------------------------------------------------------------------------------
def test_team_games(scout, synthetic_drafts):
    tg = scout.team_games("T0", None)
    raw = synthetic_drafts.drop_duplicates("game_id")
    mine = raw[(raw["team_id"] == "T0") | (raw["opponent_id"] == "T0")]
    assert len(tg) == len(mine)
    wins = mine.set_index("game_id")["game_winner_team_id"].eq("T0")
    assert (tg["won"] == wins.reindex(tg.index).astype(int)).all()
    assert len(scout.team_games("T0", 2)) < len(tg)  # patch window


def test_cards_are_valid(scout):
    tg = scout.team_games("T0", None)
    cards = scout.cards_from("T0", tg)
    assert cards
    for c in cards:
        assert c.effect >= 0 and 0 <= c.confidence <= 1
        assert all(g.games >= I.MIN_GROUP for g in c.groups)
        assert c.groups[0].rate >= c.groups[1].rate
        assert c.takeaway.endswith(".") and c.title
    side = next(c for c in cards if c.kind == "side")
    assert sum(g.games for g in side.groups) == len(tg)  # blue + red = every game
    comfort = [c for c in cards if c.kind == "comfort dependency"]
    if comfort:
        assert sum(g.games for g in comfort[0].groups) == len(tg)


def test_ban_impact_counts(scout, synthetic_drafts):
    """Banned + open games for a champion = all games where it wasn't Fearless-locked."""
    tg = scout.team_games("T1", None)
    for c in scout.ban_impact("T1", tg):
        name = c.champions[0]
        cid = scout.champ_names[scout.champ_names == name].index[0]
        banned = synthetic_drafts[
            (synthetic_drafts["action"] == "ban")
            & (synthetic_drafts["champion_id"] == cid)
            & synthetic_drafts["game_id"].isin(tg.index)
        ]["game_id"].nunique()
        assert {g.games for g in c.groups} == {banned, len(tg) - banned}


def test_phase_tells_find_planted_pattern(scout):
    """In the synthetic drafts every team picks top/jungle/mid in phase 1 and bot/support
    in phase 2, so top-mid champions must be 'early' tells and bot/support ones 'held'."""
    tells = scout.phase_tells("T0", scout.team_games("T0", None))
    assert tells
    for t in tells:
        cid = int(scout.champ_names[scout.champ_names == t.champion].index[0][1:])
        assert t.kind == ("early" if cid < 24 else "held")
        assert t.confidence > 0.9
    card = tells[0].as_card()
    assert card.kind == "phase tell" and card.unit == "picks" and card.games == tells[0].picks


def test_self_view_wording(scout):
    cards = scout.cards_from("T0", scout.team_games("T0", None), view="self")
    assert all(c.takeaway.lower().startswith(("we ", "opponents", "in games")) for c in cards)


def test_predictability(scout):
    p = scout.predictability("T0", None)
    for kind in ("pick", "ban"):
        x = p[kind]
        assert 1 <= x["effective_choices"] <= x["distinct"] + 1e-9
        assert 0 < x["top3_share"] <= 1 and len(x["top"]) <= 3
    lines = scout.predictability_takeaways("T0", p, "self")
    assert len(lines) == 2 and all(line.endswith(".") for line in lines)


# --- replication and ranking --------------------------------------------------------------
def test_replication_uses_split(scout):
    rep = I.replication(scout, splits=("2025-03-01",))
    assert {"kind", "early_effect", "late_effect", "split"} <= set(rep.columns)
    tells = rep[rep["kind"] == "phase tell"]
    assert len(tells) and (tells["late_effect"] > 0).all()  # a structural habit replicates
    summary = I.replication_summary(rep)
    assert (summary["survives"].between(0, 1)).all()
    assert summary.loc[summary["kind"] == "phase tell", "same_direction"].iloc[0] == 1
    early = scout.team_games("T0", None, before="2025-03-01")
    assert early["date"].max() < pd.Timestamp("2025-03-01")


def test_report_ranking_and_bar(scout):
    summary = pd.DataFrame(
        {
            "kind": ["phase tell", "ban impact", "side", "first pick", "comfort dependency", "fearless carry-over"],
            "early_gap": [0.4, 0.1, 0.1, 0.1, 0.1, 0.1],
            "survives": [0.6, 0.0, 0.0, 0.0, 0.5, 0.0],
        }
    )
    r = scout.report("T0", None, replication=summary)
    scores = [c.score for c in r["cards"]]
    assert scores == sorted(scores, reverse=True)
    assert 0 < len(r["top"]) <= 5 and r["passing"] >= len(r["top"])
    for c in r["top"]:
        assert c.reliability >= I.MIN_RELIABILITY and c.confidence >= I.MIN_CONFIDENCE
    assert all(c.score == 0 for c in r["cards"] if c.kind == "ban impact")
    no_rep = scout.report("T0", None)  # without replication every type counts fully
    assert all(c.reliability == 1 for c in no_rep["cards"])
