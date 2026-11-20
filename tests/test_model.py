"""Tests for draft_analyzer.model: no leakage (features and walk-forward), and
predictions on partial drafts. Synthetic data only, except the last test."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from draft_analyzer import model as M
from draft_analyzer.drafts import DRAFTS_PATH

CFG = M.FeatureConfig(elo_k=32.0)
DRAFT_ONLY = [c for c in M.FEATURES if c.startswith("d_")] + ["matchup"]


@pytest.fixture(scope="module")
def feats(synthetic_drafts):
    return M.build_features(synthetic_drafts, CFG)


def _truncate_after(drafts: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Keep game `game_id` and every game played before it (by date, series, game number)."""
    games, _ = M.prepare(drafts)
    keep = games.index[: games.index.get_loc(game_id) + 1]
    return drafts[drafts["game_id"].isin(keep)]


def _assert_no_leakage(drafts: pd.DataFrame, game_id: str, full: pd.DataFrame | None = None) -> None:
    full = M.build_features(drafts, CFG) if full is None else full
    cut = M.build_features(_truncate_after(drafts, game_id), CFG)
    key = ["game_id", "stage"]
    left = full[full["game_id"].isin(set(cut["game_id"]))].sort_values(key).reset_index(drop=True)
    right = cut.sort_values(key).reset_index(drop=True)
    assert len(left) == len(right) > 0
    pd.testing.assert_frame_equal(left, right, check_exact=False, rtol=1e-12, atol=1e-12)


# --- no leakage ---------------------------------------------------------------------------
def test_no_leakage_synthetic(synthetic_drafts, feats):
    """Features for game N don't change when every game after N is removed."""
    games, _ = M.prepare(synthetic_drafts)
    for pos in (len(games) // 4, len(games) // 2, len(games) - 2):
        _assert_no_leakage(synthetic_drafts, games.index[pos], feats)


def test_features_ignore_same_day_results(synthetic_drafts, feats):
    """A game's own result, and every other result on its date, don't move its features."""
    games, _ = M.prepare(synthetic_drafts)
    day = games["date"].iloc[len(games) // 2]
    same_day = sorted(games.index[games["date"] == day])
    flipped = synthetic_drafts.copy()
    rows = flipped["game_id"].isin(same_day)
    flipped.loc[rows, "game_winner_team_id"] = np.where(
        flipped.loc[rows, "game_winner_team_id"] == flipped.loc[rows, "team_id"],
        flipped.loc[rows, "opponent_id"],
        flipped.loc[rows, "team_id"],
    )
    a = feats.set_index(["game_id", "stage"])[M.FEATURES]
    b = M.build_features(flipped, CFG).set_index(["game_id", "stage"])[M.FEATURES]
    pd.testing.assert_frame_equal(a.loc[same_day], b.loc[same_day])


def test_walk_forward_uses_only_earlier_months(feats, monkeypatch):
    """A month's predictions, chosen C and calibration are unchanged when every later
    month is removed: each month only sees earlier games and earlier months' results."""
    monkeypatch.setattr(M, "MIN_CALIBRATION_GAMES", 50)  # synthetic data is small; make March calibrate
    full = M.walk_forward(feats, start="2025-02-01")
    cutoff = pd.Timestamp("2025-04-01")
    cut = M.walk_forward(feats[feats["date"] < cutoff], start="2025-02-01")
    assert [f["month"] for f in cut] == [f["month"] for f in full if f["month"] < cutoff]
    assert any(f["calibrated"] for f in cut)
    for a, b in zip(full, cut, strict=False):
        assert a["C"] == b["C"] and a["calibrators"] == b["calibrators"]
        pd.testing.assert_frame_equal(a["preds"], b["preds"])


def test_walk_forward_ignores_own_month_results(feats, monkeypatch):
    """Flipping every result in one month changes nothing that month predicts or chooses:
    its C and calibration come only from earlier months."""
    monkeypatch.setattr(M, "MIN_CALIBRATION_GAMES", 50)
    month = (feats["date"] >= "2025-03-01") & (feats["date"] < "2025-04-01")
    flipped = feats.assign(blue_won=np.where(month, 1 - feats["blue_won"], feats["blue_won"]))
    a = next(f for f in M.walk_forward(feats, start="2025-02-01") if f["month"] == pd.Timestamp("2025-03-01"))
    b = next(f for f in M.walk_forward(flipped, start="2025-02-01") if f["month"] == pd.Timestamp("2025-03-01"))
    assert a["calibrated"] and a["C"] == b["C"] and a["calibrators"] == b["calibrators"]
    pd.testing.assert_frame_equal(a["preds"].drop(columns="blue_won"), b["preds"].drop(columns="blue_won"))


# --- partial drafts -----------------------------------------------------------------------
def test_stage_features_build_up(feats):
    """Stage 0 has no draft signal; picks count up; a matchup needs both lane picks."""
    s0 = feats[feats["stage"] == 0]
    assert (s0[DRAFT_ONLY] == 0).all().all()
    s7 = feats[feats["stage"] == 7]  # 6 bans + the first pick
    assert (s7["d_n_picks"].abs() == 1).all() and (s7["matchup"] == 0).all()
    assert (np.sign(s7["d_n_picks"]) == np.where(s7["blue_first_pick"] == 1, 1, -1)).all()
    assert (feats.loc[feats["stage"] == 20, "d_n_picks"] == 0).all()


@pytest.fixture(scope="module")
def bundle(feats):
    return {"model": M.fit("full", feats, 0.1), "calibrator": (1.0, 0.0), "features": M.FEATURES, "config": asdict(CFG)}


@pytest.mark.parametrize("n_a,n_b", [(0, 0), (1, 0), (1, 2), (3, 2), (5, 5)])
def test_predict_partial_draft(synthetic_drafts, bundle, n_a, n_b):
    picks_a = ["C0", "C9", "C18", "C27", "C36"][:n_a]
    picks_b = ["C1", "C10", "C19", "C28", "C37"][:n_b]
    r = M.predict_draft(
        "T0", "T3", "blue", picks_a, picks_b, bans={"a": ["C2"], "b": ["C3"]}, model=bundle, drafts=synthetic_drafts
    )
    assert 0 < r["p_team_a"] < 1
    assert r["p_team_a"] + r["p_team_b"] == pytest.approx(1)
    assert r["features_blue_view"]["d_n_picks"] == n_a - n_b
    assert len(r["actions"]) == n_a + n_b + 2
    assert r["roles"]["a"] == M.ROLES[:n_a]  # synthetic picks are in role order
    if min(n_a, n_b) == 0:
        assert r["features_blue_view"]["matchup"] == 0


def test_predict_is_symmetric(synthetic_drafts, bundle):
    kw = dict(model=bundle, drafts=synthetic_drafts)
    a = M.predict_draft("T0", "T3", "blue", ["C0", "C9"], ["C1"], **kw)
    b = M.predict_draft("T3", "T0", "red", ["C1"], ["C0", "C9"], **kw)
    assert a["p_team_a"] == pytest.approx(b["p_team_b"])
    assert a["p_blue"] == pytest.approx(b["p_blue"])


def test_predict_uses_only_earlier_games(synthetic_drafts, bundle):
    """predict_draft(date=D) is unchanged when games on or after D are removed."""
    cutoff = pd.Timestamp("2025-03-01")
    earlier = synthetic_drafts[pd.to_datetime(synthetic_drafts["date"]) < cutoff]
    args = ("T0", "T3", "red", ["C0", "C9"], ["C1", "C10", "C19"])
    a = M.predict_draft(*args, date=cutoff, model=bundle, drafts=synthetic_drafts)
    b = M.predict_draft(*args, date=cutoff, model=bundle, drafts=earlier)
    assert a["p_team_a"] == pytest.approx(b["p_team_a"], abs=1e-12)


def test_explicit_roles_override_guess(synthetic_drafts, bundle):
    kw = dict(model=bundle, drafts=synthetic_drafts)
    r = M.predict_draft("T0", "T3", "blue", ["C0"], ["C1"], roles_a=["mid"], roles_b=["mid"], **kw)
    assert r["roles"] == {"a": ["mid"], "b": ["mid"]}


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(picks_a=["C0"], picks_b=["C0"]), "more than once"),
        (dict(picks_a=["C0"], bans=["C0"]), "more than once"),
        (dict(picks_a=["C0", "C1", "C2", "C3", "C4", "C5"]), "at most 5"),
        (dict(picks_a=["nope"]), "unknown champion"),
        (dict(picks_a=["C0", "C9"], roles_a=["top", "top"]), "distinct role"),
    ],
)
def test_predict_rejects_bad_drafts(synthetic_drafts, bundle, kwargs, match):
    with pytest.raises(ValueError, match=match):
        M.predict_draft("T0", "T3", "blue", model=bundle, drafts=synthetic_drafts, **kwargs)


# --- real data ----------------------------------------------------------------------------
@pytest.mark.skipif(not DRAFTS_PATH.exists(), reason="needs the local GRID drafts cache")
def test_no_leakage_real_data():
    from draft_analyzer.drafts import load_drafts

    drafts = load_drafts()
    games, _ = M.prepare(drafts)
    _assert_no_leakage(drafts, games.index[len(games) // 2])
