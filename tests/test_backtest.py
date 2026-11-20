"""Tests for the Ban plan backtest (draft_analyzer.backtest), on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from draft_analyzer import analysis as A
from draft_analyzer import backtest as B

START = "2025-03-20"


@pytest.fixture(scope="module")
def data(synthetic_drafts):
    return B.build_cases(synthetic_drafts, start=START)


def test_cases_built(data):
    c = data["cases"]
    assert len(c) > 20 and (c["prior_games"] >= B.MIN_PRIOR_GAMES).all()
    assert c["won"].isin([0, 1]).all() and c["p0"].between(0, 1).all()
    assert len(data["naive"]) and len(data["meta"])
    assert (data["naive"].groupby("case").size() <= B.N).all()


def test_plan_lists_match_recommend_bans(data, synthetic_drafts):
    """The fast vectorised lists are exactly what recommend_bans returns on earlier games."""
    names = synthetic_drafts.drop_duplicates("champion_id").set_index("champion_name")["champion_id"]
    for w in (None, B.PREVIOUS_BAN_WEIGHTS):
        lists = B.plan_lists(data, w)
        for case in data["cases"].sample(5, random_state=0).itertuples():
            prior = synthetic_drafts[pd.to_datetime(synthetic_drafts["date"]) < case.date]
            locked = set(data["locked"].loc[data["locked"]["case"] == case.case, "champion_id"])
            ref = A.recommend_bans(case.us, case.them, n=B.N, weights=w, drafts=prior,
                                   last_patches=B.LAST_PATCHES, exclude=locked)
            mine = lists[lists["case"] == case.case].sort_values("rank")["champion_id"].tolist()
            assert mine == [names[c] for c in ref["champion"]]


def test_no_leakage(synthetic_drafts, data):
    """Cases (stats, baselines) for games before a cutoff are unchanged when every later
    game is removed."""
    cutoff = pd.Timestamp("2025-04-10")
    cut = B.build_cases(synthetic_drafts[pd.to_datetime(synthetic_drafts["date"]) < cutoff], start=START)
    keep = data["cases"].loc[data["cases"]["date"] < cutoff, "case"]
    pd.testing.assert_frame_equal(data["cases"][data["cases"]["case"].isin(keep)].reset_index(drop=True),
                                  cut["cases"].reset_index(drop=True))
    for key in ("stats", "naive", "meta"):
        a = data[key][data[key]["case"].isin(keep)].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, cut[key].reset_index(drop=True))


def test_evaluate_counts():
    """Hand-made case: 3 listed champions, one banned by us, one picked by us, one left
    available and picked by them."""
    data = {"cases": pd.DataFrame({"case": [0, 1], "won": [1, 0], "p0": [0.5, 0.5],
                                   "game_id": ["g0", "g1"], "date": pd.to_datetime(["2026-01-01"] * 2)}),
            "facts": pd.DataFrame({"case": [0, 0, 0, 1], "champion_id": ["a", "b", "c", "x"],
                                   "ban_by_us": [True, False, False, False], "ban_by_them": [False] * 4,
                                   "pick_by_us": [False, True, False, False],
                                   "pick_by_them": [False, False, True, False]})}
    lists = pd.DataFrame({"case": [0, 0, 0, 1], "champion_id": ["a", "b", "c", "y"], "rank": [1, 2, 3, 1]})
    m = B.evaluate(data, lists)
    assert m["available"] == 2 and m["picked"] == 1 and m["pick_rate"] == 0.5     # c (case 0) + y (case 1)
    assert m["games"] == 2 and m["game_hits"] == 1
    assert m["got_games"] == 1 and m["win_rate_got"] == 1 and m["excess_got"] == pytest.approx(0.5)
    assert m["we_banned"] == pytest.approx(1 / 4)


def test_paired_gain_and_weights(data):
    test = pd.Index(data["cases"]["case"])
    lists = B.plan_lists(data)
    same = B.paired_gain(data, lists, lists, test, n_boot=200)
    assert same["diff"] == 0 and same["lo"] == 0 and same["hi"] == 0
    for k in B.GRID:
        assert A.DEFAULT_BAN_WEIGHTS[k] in B.GRID[k]          # the defaults are a grid point
    assert any(B.PREVIOUS_BAN_WEIGHTS[k] != A.DEFAULT_BAN_WEIGHTS[k] for k in B.PREVIOUS_BAN_WEIGHTS)
    assert np.isfinite(B.evaluate(data, lists)["pick_rate"])
