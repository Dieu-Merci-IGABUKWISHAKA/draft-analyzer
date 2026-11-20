"""Tests for Match prep (draft_analyzer.prep), the ordered ban reasons and the Ban plan
caption. Synthetic data only."""

from __future__ import annotations

import re
from dataclasses import asdict

import pandas as pd
import pytest

from draft_analyzer import analysis as A
from draft_analyzer import backtest as B
from draft_analyzer import insights as I
from draft_analyzer import model as M
from draft_analyzer import prep as P
from draft_analyzer import simulator as S

CFG = M.FeatureConfig(elo_k=32.0)


def test_ban_reasons_ordered_by_contribution(synthetic_drafts):
    recs = A.recommend_bans("T0", "T3", 5, drafts=synthetic_drafts, last_patches=6, opponent_name="Team 3")
    assert len(recs)
    for r in recs.itertuples():
        sizes = [abs(v) for _, v in r.drivers]
        assert sizes == sorted(sizes, reverse=True)
        assert len(r.drivers) == len(r.reason_lines)
        for (kind, _), line in zip(r.drivers, r.reason_lines, strict=False):
            expected = {
                "presence": "Picked",
                "win rate": "Win rate",
                "banned by others": "Banned vs Team 3",
                "we play it": "We pick it",
            }[kind]
            assert line.startswith(expected)
            assert not re.search(r"\b1 games", line)
    # the ranking itself is unchanged by the new columns
    plain = A.recommend_bans("T0", "T3", 5, drafts=synthetic_drafts, last_patches=6)
    assert list(plain["champion"]) == list(recs["champion"])


def test_headline_caption():
    assert B.headline_caption() == (
        "Tested on 2026 games it never saw: opponents picked a listed champion 34% of "
        "the time when left open, vs 27% for the meta top 5."
    )


@pytest.fixture(scope="module")
def prep(synthetic_drafts):
    feats = M.build_features(synthetic_drafts, CFG)
    bundle = {
        "model": M.fit("full", feats, 0.1),
        "calibrator": (1.0, 0.0),
        "features": M.FEATURES,
        "config": asdict(CFG),
        "bootstrap": M.bootstrap_models(feats, 0.1, n=4),
    }
    live = M.live_context(bundle, synthetic_drafts)
    scout = I.Scout(synthetic_drafts)
    acc = pd.DataFrame(
        {
            "kind": ["pick", "pick", "ban", "ban"],
            "method": ["tendency", "meta only"] * 2,
            "predictions": [100] * 4,
            "games": [10] * 4,
            "top1": [0.1] * 4,
            "top5": [0.44, 0.37, 0.38, 0.26],
        }
    )
    return P.match_prep(
        "T0",
        "T3",
        drafts=synthetic_drafts,
        scout=scout,
        sim=S.Simulator("T0", "T3", live),
        insight_rep=None,
        likely_acc=acc,
    )


def test_match_prep_contents(prep, synthetic_drafts):
    assert prep["our"] == "Team 0" and prep["opp"] == "Team 3"
    assert len(prep["likely"]) == 5 and all(0 < x["prob"] < 1 for x in prep["likely"])
    assert 1 <= len(prep["bans"]) <= 5 and all(b["lead"] for b in prep["bans"])
    assert len(prep["beat"]) <= 3
    beat_titles = {c["title"] for c in prep["beat"]}
    assert not beat_titles & {t["title"] for t in prep["tells"]}  # nothing shown twice
    h2h = A.record("T0", None, vs="T3", drafts=synthetic_drafts)
    assert prep["h2h"]["all"] == h2h
    f = prep["form"]["them"]
    assert f["last_n"] == len(f["last"]) <= P.FORM_GAMES and set(f["last"]) <= {"W", "L"}
    assert prep["proof"] == "Tendencies tested on 10 unseen games: next pick in our top 5 44% vs 37% for the meta."
    assert set(prep["tabs"].values()) >= {"Ban plan", "Insights", "Draft simulator", "Against us"}


def test_render_html(prep):
    p = dict(prep, opp="<Team & 3>")
    doc = P.render_html(p)
    assert doc.startswith("<!doctype html>") and "&lt;Team &amp; 3&gt;" in doc and "<Team & 3>" not in doc
    assert "<img" not in doc and "<script" not in doc
    for heading in (
        "Head-to-head and form",
        "What they'll draft",
        "Our ban plan",
        "How to beat them",
        "what they ban against us",
        "our predictable habits",
    ):
        assert heading in doc
    assert prep["ban_proof"] in doc and "@media print" in doc
