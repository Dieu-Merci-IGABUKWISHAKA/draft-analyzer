"""Tests for draft_analyzer.simulator, on the synthetic drafts table."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from draft_analyzer import model as M
from draft_analyzer import simulator as S
from draft_analyzer.analysis import _bucket

CFG = M.FeatureConfig(elo_k=32.0)


@pytest.fixture(scope="module")
def live(synthetic_drafts):
    feats = M.build_features(synthetic_drafts, CFG)
    bundle = {
        "model": M.fit("full", feats, 0.1),
        "calibrator": (1.0, 0.0),
        "features": M.FEATURES,
        "config": asdict(CFG),
        "bootstrap": M.bootstrap_models(feats, 0.1, n=8),
    }
    return M.live_context(bundle, synthetic_drafts)


@pytest.fixture(scope="module")
def sim(live):
    return S.Simulator("T0", "T3", live)


def free(sim, setup, acts, role):
    """An available synthetic champion of `role` (C0-C7 top, C8-C15 jungle, ...)."""
    r = S.ROLES.index(role)
    return next(c for c in (f"C{r * 8 + j}" for j in range(8)) if c not in sim.unavailable(setup, acts))


def fill(sim, setup, n):
    """Fill the first n slots: our best option on our turn, their most likely choice on theirs."""
    acts = ()
    for _ in range(n):
        best = sim.best_options(setup, acts)
        choice = best.iloc[0] if best is not None and len(best) else sim.likely_next(setup, acts)["table"].iloc[0]
        acts = sim.apply(setup, acts, choice["champion_id"], choice["role"])
    return acts


# --- board --------------------------------------------------------------------------------
def test_slot_order_matches_pro_draft():
    s = S.slots(S.Setup(we_first=True))
    assert [x["who"] for x in s] == (
        ["us", "them"] * 3
        + ["us", "them", "them", "us", "us", "them"]
        + ["them", "us", "them", "us"]
        + ["them", "us", "us", "them"]
    )
    assert [x["label"] for x in s][:7] == ["Ban 1", "Ban 2", "Ban 3", "Ban 4", "Ban 5", "Ban 6", "Pick 1"]
    assert s[12]["label"] == "Ban 7" and s[16]["label"] == "Pick 7" and s[19]["label"] == "Pick 10"
    assert [x["phase"] for x in s if x["kind"] == "pick"] == [1] * 6 + [2] * 4
    flipped = S.slots(S.Setup(we_first=False))
    assert all(a["who"] != b["who"] for a, b in zip(s, flipped, strict=False))
    red = S.slots(S.Setup(our_side="red"))
    assert red[0]["side"] == "red" and red[1]["side"] == "blue"


def test_apply_enforces_rules(sim):
    setup = S.Setup(locked=frozenset({"C5"}))
    with pytest.raises(ValueError, match="Fearless"):
        sim.apply(setup, (), "C5")
    acts = sim.apply(setup, (), "C6")
    assert acts[0].role is None  # bans have no role
    with pytest.raises(ValueError, match="already picked or banned"):
        sim.apply(setup, acts, "C6")
    acts = fill(sim, setup, 6)
    acts = sim.apply(setup, acts, free(sim, setup, acts, "top"), "top")  # our first pick
    acts = sim.apply(setup, acts, free(sim, setup, acts, "top"), "top")  # their picks
    acts = sim.apply(setup, acts, free(sim, setup, acts, "jungle"), "jungle")
    with pytest.raises(ValueError, match="already filled"):
        sim.apply(setup, acts, free(sim, setup, acts, "top"), "top")  # our top is taken
    full = fill(sim, S.Setup(), 20)
    with pytest.raises(ValueError, match="complete"):
        sim.apply(S.Setup(), full, "C39")


def test_set_role_swaps_teammates(sim):
    setup = S.Setup()
    acts = fill(sim, setup, 6)
    for role in ("top", "top", "jungle", "mid"):  # slots 7 (us), 8-9 (them), 10 (us)
        acts = sim.apply(setup, acts, free(sim, setup, acts, role), role)
    acts = sim.set_role(setup, acts, 10, "top")
    roles = {a.slot: a.role for a in acts}
    assert roles[10] == "top" and roles[7] == "mid"


# --- win chance ---------------------------------------------------------------------------
def test_win_chance_band_and_trajectory(sim):
    setup = S.Setup()
    acts = fill(sim, setup, 12)
    wc = sim.win_chance(setup, acts)
    assert 0 < wc["p"] < 1 and wc["lo"] <= wc["p"] <= wc["hi"]
    tr = sim.trajectory(setup, acts)
    assert len(tr) == len(acts) + 1
    assert tr["p"].iloc[-1] == pytest.approx(wc["p"])
    assert tr["p"].iloc[0] == pytest.approx(sim.win_chance(setup, ())["p"])
    assert tr["p"].iloc[:7].nunique() == 1  # bans don't enter the win model


def test_sides_are_consistent(sim):
    """Our chance as blue equals one minus theirs when the same draft is seen from their side."""
    a = S.Simulator("T0", "T3", sim.live)
    b = S.Simulator("T3", "T0", sim.live)
    acts_a = fill(a, S.Setup(our_side="blue", we_first=True), 12)
    acts_b = tuple(acts_a)  # same slots: in b, "us" is T3 and moves second, on red
    pa = a.win_chance(S.Setup(our_side="blue", we_first=True), acts_a)["p"]
    pb = b.win_chance(S.Setup(our_side="red", we_first=False), acts_b)["p"]
    assert pa == pytest.approx(1 - pb)


# --- best options -------------------------------------------------------------------------
def test_best_pick_shift_matches_applying_it(sim):
    setup = S.Setup()
    acts = fill(sim, setup, 6)  # our first pick is next
    best = sim.best_options(setup, acts)
    assert len(best) == 5 and best["shift_pts"].is_monotonic_decreasing
    now = sim.win_chance(setup, acts)["p"]
    for r in best.itertuples():
        after = sim.win_chance(setup, sim.apply(setup, acts, r.champion_id, r.role))["p"]
        assert (after - now) * 100 == pytest.approx(r.shift_pts, abs=1e-9)
        assert r.lo_pts <= r.hi_pts
    assert set(best["reason"]) <= set(S.REASONS) | {"no single strong reason"}


def test_best_options_respect_board(sim):
    setup = S.Setup(locked=frozenset({"C0", "C9"}))
    acts = fill(sim, setup, 6)
    best = sim.best_options(setup, acts, n=40)
    assert not set(best["champion_id"]) & sim.unavailable(setup, acts)
    acts = sim.apply(setup, acts, best["champion_id"].iloc[0], best["role"].iloc[0])
    assert sim.best_options(setup, acts) is None  # their turn
    acts = fill_from(sim, setup, acts, 9)  # through slot 9: our picks 10-11 next
    best = sim.best_options(setup, acts, n=40)
    assert set(best["role"]) <= set(sim.open_roles(setup, acts, "us"))


def fill_from(sim, setup, acts, upto):
    while len(acts) < upto:
        best = sim.best_options(setup, acts)
        c = best.iloc[0] if best is not None and len(best) else sim.likely_next(setup, acts)["table"].iloc[0]
        acts = sim.apply(setup, acts, c["champion_id"], c["role"])
    return acts


def test_ban_options(sim):
    setup = S.Setup()
    best = sim.best_options(setup, ())  # our first ban
    assert len(best) == 5 and best["role"].notna().all()
    assert ((best["p_them"] > 0) & (best["p_them"] <= 1)).all()
    assert best["shift_pts"].is_monotonic_decreasing
    assert best["evidence"].str.startswith("They pick it").all()


# --- likely from them ---------------------------------------------------------------------
def test_likely_next(sim):
    setup = S.Setup(locked=frozenset({"C1"}))
    ln = sim.likely_next(setup, ())
    assert ln["kind"] == "ban" and ln["slot"]["slot"] == 2
    acts = fill(sim, setup, 7)  # their two picks are next
    ln = sim.likely_next(setup, acts)
    t = ln["table"]
    assert ln["kind"] == "pick" and len(t) == 5
    assert t["prob"].is_monotonic_decreasing and t["prob"].sum() <= 1 + 1e-9
    assert not set(t["champion_id"]) & sim.unavailable(setup, acts)
    assert set(t["role"]) <= set(sim.open_roles(setup, acts, "them"))
    assert sim.likely_next(setup, fill(sim, setup, 20)) is None


def test_tendencies_use_only_earlier_games(synthetic_drafts, live):
    cutoff = pd.Timestamp("2025-03-01")
    _, earlier = M.prepare(synthetic_drafts[pd.to_datetime(synthetic_drafts["date"]) < cutoff])
    a = S.tendency_tables(live.actions, cutoff, live.names.index)
    b = S.tendency_tables(earlier, cutoff, live.names.index)
    for team in ("T0", "T3"):
        for k, v in a.team(team).items():
            np.testing.assert_array_equal(v, b.team(team)[k])
    for ph in (1, 2):
        np.testing.assert_array_equal(a.meta_pick[ph], b.meta_pick[ph])


def test_likely_accuracy_runs(live):
    acc = S.likely_accuracy(live, start="2025-04-01")
    assert set(acc["kind"]) == {"pick", "ban"} and (acc["predictions"] > 0).all()
    assert ((acc["top1"] <= acc["top5"]) & (acc["top5"] <= 1)).all()


def test_step_is_fast(sim):
    setup = S.Setup()
    acts = fill(sim, setup, 15)
    t = time.perf_counter()
    (
        sim.win_chance(setup, acts),
        sim.trajectory(setup, acts),
        sim.best_options(setup, acts),
        sim.likely_next(setup, acts),
    )
    assert time.perf_counter() - t < 1.0


def test_fearless_buckets_split_g3_g5():
    assert list(_bucket(pd.Series([1, 2, 3, 4, 5]))) == ["G1", "G2", "G3", "G4", "G5"]
