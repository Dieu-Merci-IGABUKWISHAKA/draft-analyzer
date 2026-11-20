"""Tests for draft_analyzer.model: no leakage, and predictions on partial drafts."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from draft_analyzer import model as M
from draft_analyzer.drafts import DRAFTS_PATH

CFG = M.FeatureConfig(elo_k=32.0)


def _truncate_after(drafts: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Keep game `game_id` and every game played before it (by date, series, game number)."""
    games, _ = M.prepare(drafts)
    keep = games.index[: games.index.get_loc(game_id) + 1]
    return drafts[drafts["game_id"].isin(keep)]


def _assert_no_leakage(drafts: pd.DataFrame, game_id: str) -> None:
    full = M.build_features(drafts, CFG)
    cut = M.build_features(_truncate_after(drafts, game_id), CFG)
    key = ["game_id", "stage"]
    left = full[full["game_id"].isin(set(cut["game_id"]))].sort_values(key).reset_index(drop=True)
    right = cut.sort_values(key).reset_index(drop=True)
    assert len(left) == len(right) > 0
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


def test_no_leakage_synthetic(synthetic_drafts):
    """Features for game N don't change when every game after N is removed."""
    games, _ = M.prepare(synthetic_drafts)
    for pos in (len(games) // 3, len(games) // 2, len(games) - 2):
        _assert_no_leakage(synthetic_drafts, games.index[pos])


def test_features_ignore_same_day_results(synthetic_drafts):
    """A game's own result, and other results on its date, don't move its features."""
    games, _ = M.prepare(synthetic_drafts)
    target = games.index[len(games) // 2]
    day = games.loc[target, "date"]
    flipped = synthetic_drafts.copy()
    same_day = set(games.index[games["date"] == day])
    rows = flipped["game_id"].isin(same_day)
    flipped.loc[rows, "game_winner_team_id"] = np.where(
        flipped.loc[rows, "game_winner_team_id"] == flipped.loc[rows, "team_id"],
        flipped.loc[rows, "opponent_id"], flipped.loc[rows, "team_id"])
    a = M.build_features(synthetic_drafts, CFG).set_index(["game_id", "stage"])[M.FEATURES]
    b = M.build_features(flipped, CFG).set_index(["game_id", "stage"])[M.FEATURES]
    pd.testing.assert_frame_equal(a.loc[sorted(same_day)], b.loc[sorted(same_day)])


def test_partial_draft_stage_sums(synthetic_drafts):
    """Stage 0 has no draft signal; the draft features build up as slots fill."""
    f = M.build_features(synthetic_drafts, CFG)
    s0 = f[f["stage"] == 0]
    assert (s0[[c for c in M.FEATURES if c.startswith("d_")]] == 0).all().all()
    s7 = f[f["stage"] == 7]   # 6 bans + the first pick
    assert (s7["d_n_picks"] == 1).all() and (s7["d_n_bans"] == 0).all()
    assert (f[f["stage"] == 20]["d_n_picks"] == 0).all()


@pytest.fixture(scope="module")
def bundle(synthetic_drafts):
    feats = M.build_features(synthetic_drafts, CFG)
    model = M.fit(M.make_model("lr"), feats, M.FEATURES)
    return {"model": model, "kind": "lr", "features": M.FEATURES, "config": asdict(CFG)}


@pytest.mark.parametrize("n_a,n_b", [(0, 0), (1, 0), (1, 2), (3, 2), (5, 5)])
def test_predict_partial_draft(synthetic_drafts, bundle, n_a, n_b):
    picks_a = ["C0", "C9", "C18", "C27", "C36"][:n_a]
    picks_b = ["C1", "C10", "C19", "C28", "C37"][:n_b]
    r = M.predict_draft("T0", "T3", "blue", picks_a, picks_b, bans={"a": ["C2"], "b": ["C3"]},
                        model=bundle, drafts=synthetic_drafts)
    assert 0 < r["p_team_a"] < 1
    assert r["p_team_a"] + r["p_team_b"] == pytest.approx(1)
    assert r["features_first_pick_view"]["d_n_picks"] == n_a - n_b
    assert len(r["actions"]) == n_a + n_b + 2


def test_predict_is_symmetric(synthetic_drafts, bundle):
    kw = dict(model=bundle, drafts=synthetic_drafts)
    a = M.predict_draft("T0", "T3", "blue", ["C0", "C9"], ["C1"], **kw)
    b = M.predict_draft("T3", "T0", "red", ["C1"], ["C0", "C9"], **kw)
    assert a["p_team_a"] == pytest.approx(b["p_team_b"])


def test_predict_uses_only_earlier_games(synthetic_drafts, bundle):
    """predict_draft(date=D) is unchanged when games on or after D are removed."""
    cutoff = pd.Timestamp("2025-02-01")
    earlier = synthetic_drafts[pd.to_datetime(synthetic_drafts["date"]) < cutoff]
    args = ("T0", "T3", "red", ["C0", "C9"], ["C1", "C10", "C19"])
    a = M.predict_draft(*args, date=cutoff, model=bundle, drafts=synthetic_drafts)
    b = M.predict_draft(*args, date=cutoff, model=bundle, drafts=earlier)
    assert a["p_team_a"] == pytest.approx(b["p_team_a"], abs=1e-12)


@pytest.mark.parametrize("kwargs,match", [
    (dict(picks_a=["C0"], picks_b=["C0"]), "more than once"),
    (dict(picks_a=["C0"], bans=["C0"]), "more than once"),
    (dict(picks_a=["C0", "C1", "C2", "C3", "C4", "C5"]), "at most 5"),
    (dict(picks_a=["nope"]), "unknown champion"),
])
def test_predict_rejects_bad_drafts(synthetic_drafts, bundle, kwargs, match):
    with pytest.raises(ValueError, match=match):
        M.predict_draft("T0", "T3", "blue", model=bundle, drafts=synthetic_drafts, **kwargs)


@pytest.mark.skipif(not DRAFTS_PATH.exists(), reason="needs the local GRID drafts cache")
def test_no_leakage_real_data():
    from draft_analyzer.drafts import load_drafts
    drafts = load_drafts()
    games, _ = M.prepare(drafts)
    _assert_no_leakage(drafts, games.index[len(games) // 2])
