"""Walk-forward backtest of the Ban plan (analysis.recommend_bans).

For every game and each side: the team we scout is "them", the other team is "us", and
the Ban plan is computed from games strictly before that game's date (dashboard defaults:
each team's last 6 patches, half-life 3). Champions already used earlier in the series
(hard Fearless) are excluded from every list. Three top-5 lists are compared:

  ban plan   recommend_bans (default weights, or weights tuned on 2025 games)
  naive      their 5 most-picked champions in all earlier games
  meta       the 5 most-picked champions league-wide in the last 6 patches

Metrics on the champions in a list that we did NOT ban and that were still available to
them (not banned by either team, not picked by us, not Fearless-locked):
  pick rate      share of those champions they picked
  game hit rate  share of games where they picked at least one of them
  win check      their win rate in games where they got a listed champion vs their
                 overall rate and vs their pre-game rate (the excess is "wins above
                 expectation" that banning those champions could have taken away)

Weights are tuned on 2025 games only, by pick rate (Phase 3 showed that win-rate gaps
per champion mostly don't replicate, so the reliably measurable job of a ban is to deny
what they would actually play). The tuned weights replace the defaults only if they beat
them on 2025 games; 2026 games are the test.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer import analysis as A  # noqa: E402
from draft_analyzer import insights as I  # noqa: E402
from draft_analyzer.config import CACHE_DIR  # noqa: E402

TUNE_START, TEST_START = "2025-01-01", "2026-01-01"
LAST_PATCHES = 6  # dashboard default window
MIN_PRIOR_GAMES = 10  # the scouted team needs this many games in its window
N = 5
# The first grid (pick up to 2, banned_by_others up to 1, prior_games 5/10/20) found its
# best 2025 setting on the grid's edge, and prior_games changed the result by < 0.2 pts.
# So the grid was widened and prior_games fixed at 10. Both decisions used 2025 games only.
GRID = {
    "win_rate": [0.0, 1.0, 3.0],
    "pick": [0.0, 0.5, 1.0, 2.0, 4.0, 8.0],
    "banned_by_others": [0.0, 0.25, 1.0, 2.0, 4.0],
    "our_pick_penalty": [0.0, 0.5],
    "prior_games": [10.0],
    "min_picks": [1, 2, 4],
}
CASES_PATH = CACHE_DIR / "ban_backtest_cases.joblib"
# Headline 2026 result of the last full run (python src/draft_analyzer/backtest.py), shown
# in the app; docs/VALIDATION.md has the full tables. Update both after a re-run.
HEADLINE = {"plan_pick_rate": 0.338, "meta_pick_rate": 0.269, "naive_pick_rate": 0.198, "test_games": 439}


def headline_caption() -> str:
    h = HEADLINE
    return (
        f"Tested on 2026 games it never saw: opponents picked a listed champion {h['plan_pick_rate']:.0%} of the "
        f"time when left open, vs {h['meta_pick_rate']:.0%} for the meta top 5."
    )


# The hand-set weights used before this backtest (Phases 0-4); the tuned ones are now
# analysis.DEFAULT_BAN_WEIGHTS.
PREVIOUS_BAN_WEIGHTS = {
    "win_rate": 3.0,
    "pick": 0.0,
    "banned_by_others": 0.25,
    "our_pick_penalty": 0.5,
    "prior_games": 10.0,
    "min_picks": 4,
}


# --- cases --------------------------------------------------------------------------------
def build_cases(
    drafts: pd.DataFrame,
    start: str = TUNE_START,
    end: str | None = None,
    last_patches: int = LAST_PATCHES,
    half_life: float = A.DEFAULT_BAN_WEIGHTS["half_life"],
) -> dict:
    """Everything the evaluation needs, computed once (the slow part):
    cases   one row per (game, scouted team): game, them, us, won, p0 (their pre-game win rate), date
    stats   ban_stats rows per case (champion candidates) with the case's W / opp_wr / ours_W
    facts   per case, what happened to each champion in the game (banned by us / them, picked by us / them)
    locked  per case, champions used earlier in the series
    naive, meta  baseline lists (case, champion_id, rank)"""
    d = drafts.copy()
    d["date"] = pd.to_datetime(d["date"])
    for col in ("action", "patch"):
        d[col] = d[col].astype(str)
    games = d.drop_duplicates(["game_id", "team_id"])[
        ["game_id", "series_id", "game_number", "date", "team_id", "opponent_id", "game_winner_team_id"]
    ]
    games = games[(games["date"] >= pd.Timestamp(start)) & ((games["date"] < pd.Timestamp(end)) if end else True)]
    picks_all = d[d["action"] == "pick"]

    cases, stats, facts, locked, naive, meta = [], [], [], [], [], []
    for day, todays in games.groupby("date"):
        prior = drafts[pd.to_datetime(drafts["date"]) < day]
        pp = prior[prior["action"].astype(str) == "pick"]
        patches = sorted(prior["patch"].dropna().astype(str).unique(), key=A._patch_key)[-last_patches:]
        meta_counts = pp[pp["patch"].astype(str).isin(patches)]["champion_id"].value_counts()
        for r in todays.itertuples(index=False):
            them, us = r.team_id, r.opponent_id
            st, ctx = A.ban_stats(us, them, prior, last_patches, half_life)
            if len(ctx["games"]) < MIN_PRIOR_GAMES:
                continue
            cid = len(cases)
            cases.append(
                {
                    "case": cid,
                    "game_id": r.game_id,
                    "them": them,
                    "us": us,
                    "date": day,
                    "won": int(r.game_winner_team_id == them),
                    "p0": float(ctx["opp_wr"]),
                    "prior_games": len(ctx["games"]),
                }
            )
            stats.append(
                st.assign(case=cid, W=ctx["W"], opp_wr=ctx["opp_wr"], ours_W=ctx["ours_W"])
                .rename_axis("champion_id")
                .reset_index()
            )
            game = d[d["game_id"] == r.game_id]
            for (team, action), g in game.groupby(["team_id", "action"]):
                who = "us" if team == us else "them"
                facts.append(pd.DataFrame({"case": cid, "champion_id": g["champion_id"], "fact": f"{action}_by_{who}"}))
            used = picks_all[(picks_all["series_id"] == r.series_id) & (picks_all["game_number"] < r.game_number)]
            lk = set(used["champion_id"])
            locked.append(pd.DataFrame({"case": cid, "champion_id": sorted(lk)}))
            own = pp[pp["team_id"] == them]["champion_id"].value_counts()
            naive.append(_top(own, lk, cid))
            meta.append(_top(meta_counts, lk, cid))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()  # noqa: E731
    f = cat(facts)
    facts_wide = (
        f.assign(v=True).pivot_table(
            index=["case", "champion_id"], columns="fact", values="v", aggfunc="any", fill_value=False
        )
        if len(f)
        else pd.DataFrame()
    )
    for col in ("ban_by_us", "ban_by_them", "pick_by_us", "pick_by_them"):
        if col not in facts_wide:
            facts_wide[col] = False
    return {
        "cases": pd.DataFrame(cases),
        "stats": cat(stats),
        "facts": facts_wide.reset_index(),
        "locked": cat(locked),
        "naive": cat(naive),
        "meta": cat(meta),
    }


def _top(counts: pd.Series, locked: set, case: int, n: int = N) -> pd.DataFrame:
    top = [c for c in counts.index if c not in locked][:n]
    return pd.DataFrame({"case": case, "champion_id": top, "rank": range(1, len(top) + 1)})


def load_cases(drafts: pd.DataFrame | None = None, rebuild: bool = False) -> dict:
    """build_cases over all games from TUNE_START, cached in data_cache/ (row-level: gitignored)."""
    import joblib

    from draft_analyzer.drafts import load_drafts

    if not rebuild and CASES_PATH.exists():
        return joblib.load(CASES_PATH)
    data = build_cases(load_drafts() if drafts is None else drafts)
    CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, CASES_PATH)
    return data


# --- lists and metrics --------------------------------------------------------------------
def plan_lists(data: dict, weights: dict | None = None, n: int = N) -> pd.DataFrame:
    """The Ban plan's top-n per case under `weights`, scored with analysis.ban_scores."""
    w = {**A.DEFAULT_BAN_WEIGHTS, **(weights or {})}
    st = data["stats"]
    if st.empty:
        return pd.DataFrame(columns=["case", "champion_id", "rank"])
    ctx = {"W": st["W"].to_numpy(), "opp_wr": st["opp_wr"].to_numpy(), "ours_W": st["ours_W"].to_numpy()}
    scored = A.ban_scores(st, ctx, w)
    lk = data["locked"]
    is_locked = (
        pd.MultiIndex.from_frame(scored[["case", "champion_id"]]).isin(
            pd.MultiIndex.from_frame(lk[["case", "champion_id"]])
        )
        if len(lk)
        else np.zeros(len(scored), bool)
    )
    keep = scored[(scored["picks"] >= w["min_picks"]) & ~is_locked]
    top = keep.sort_values(["case", "score"], ascending=[True, False], kind="stable").groupby("case").head(n)
    return top.assign(rank=top.groupby("case").cumcount() + 1)[["case", "champion_id", "rank"]]


def evaluate(data: dict, lists: pd.DataFrame, cases: pd.Index | None = None) -> dict:
    """Metrics for one set of lists (case, champion_id) over `cases` (default: all)."""
    c = data["cases"].set_index("case")
    if cases is not None:
        c = c.loc[c.index.intersection(cases)]
    L = lists[lists["case"].isin(c.index)].merge(data["facts"], on=["case", "champion_id"], how="left")
    for col in ("ban_by_us", "ban_by_them", "pick_by_us", "pick_by_them"):
        L[col] = L[col].fillna(False).astype(bool)
    not_ours = L[~L["ban_by_us"]]
    avail = not_ours[~not_ours["ban_by_them"] & ~not_ours["pick_by_us"]]
    k, n = int(avail["pick_by_them"].sum()), len(avail)
    per_game = avail.groupby("case")["pick_by_them"].any().reindex(c.index)
    games_with_avail = per_game.dropna()
    got = games_with_avail[games_with_avail].index
    won_got, won_all = c.loc[got, "won"], c["won"]
    lo, hi = I.wilson(k, n)
    glo, ghi = I.wilson(int(games_with_avail.sum()), len(games_with_avail))
    wlo, whi = I.wilson(int(won_got.sum()), len(won_got))
    return {
        "cases": len(c),
        "listed": len(L),
        "we_banned": float(L["ban_by_us"].mean()) if len(L) else np.nan,
        "available": n,
        "picked": k,
        "pick_rate": k / n if n else np.nan,
        "pick_lo": lo,
        "pick_hi": hi,
        "games": len(games_with_avail),
        "game_hits": int(games_with_avail.sum()),
        "game_hit_rate": float(games_with_avail.mean()) if len(games_with_avail) else np.nan,
        "game_lo": glo,
        "game_hi": ghi,
        "got_games": len(won_got),
        "got_wins": int(won_got.sum()),
        "win_rate_got": float(won_got.mean()) if len(won_got) else np.nan,
        "win_lo": wlo,
        "win_hi": whi,
        "win_rate_all": float(won_all.mean()),
        "excess_got": float((c.loc[got, "won"] - c.loc[got, "p0"]).mean()) if len(got) else np.nan,
    }


def paired_gain(
    data: dict, a: pd.DataFrame, b: pd.DataFrame, cases: pd.Index, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Pick-rate difference (list a minus list b) with a 90% bootstrap interval over games."""

    def per_case(lists: pd.DataFrame) -> pd.DataFrame:
        L = lists[lists["case"].isin(cases)].merge(data["facts"], on=["case", "champion_id"], how="left")
        for col in ("ban_by_us", "ban_by_them", "pick_by_us", "pick_by_them"):
            L[col] = L[col].fillna(False).astype(bool)
        av = L[~L["ban_by_us"] & ~L["ban_by_them"] & ~L["pick_by_us"]]
        return av.groupby("case")["pick_by_them"].agg(["sum", "size"]).reindex(cases).fillna(0)

    pa, pb = per_case(a), per_case(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(cases), size=(n_boot, len(cases)))
    ka, na, kb, nb = (x.to_numpy()[idx].sum(axis=1) for x in (pa["sum"], pa["size"], pb["sum"], pb["size"]))
    diffs = ka / na - kb / nb
    point = pa["sum"].sum() / pa["size"].sum() - pb["sum"].sum() / pb["size"].sum()
    return {"diff": float(point), "lo": float(np.percentile(diffs, 5)), "hi": float(np.percentile(diffs, 95))}


# --- tuning and the report ----------------------------------------------------------------
def tune(data: dict, train: pd.Index) -> pd.DataFrame:
    """Every GRID combination scored by pick rate on the training cases (2025)."""
    rows = []
    keys = list(GRID)
    for combo in itertools.product(*GRID.values()):
        w = dict(zip(keys, combo, strict=False))
        m = evaluate(data, plan_lists(data, w), train)
        rows.append(
            {**w, "pick_rate": m["pick_rate"], "game_hit_rate": m["game_hit_rate"], "excess_got": m["excess_got"]}
        )
    return pd.DataFrame(rows).sort_values("pick_rate", ascending=False).reset_index(drop=True)


def report(data: dict | None = None) -> dict:
    data = load_cases() if data is None else data
    c = data["cases"]
    train = pd.Index(c.loc[c["date"] < pd.Timestamp(TEST_START), "case"])
    test = pd.Index(c.loc[c["date"] >= pd.Timestamp(TEST_START), "case"])
    grid = tune(data, train)
    best = {k: grid.iloc[0][k] for k in GRID}
    best["min_picks"] = int(best["min_picks"])
    previous = dict(PREVIOUS_BAN_WEIGHTS)
    m_prev_train = evaluate(data, plan_lists(data, previous), train)
    adopt = grid.iloc[0]["pick_rate"] > m_prev_train["pick_rate"]  # decided on 2025 only
    in_use = all(float(A.DEFAULT_BAN_WEIGHTS[k]) == float(best[k]) for k in GRID)
    lists = {
        "Ban plan (previous weights)": plan_lists(data, previous),
        "Ban plan (tuned on 2025)": plan_lists(data, best),
        "Naive: their 5 most-picked": data["naive"],
        "League meta: top 5 picks": data["meta"],
    }
    out = {
        "train_cases": len(train),
        "test_cases": len(test),
        "tuned": {k: float(v) for k, v in best.items()},
        "previous": previous,
        "adopt_tuned": bool(adopt),
        "tuned_are_defaults": in_use,
        "grid_top": grid.head(10),
        "grid_size": len(grid),
        "test_games": int(c.loc[c["case"].isin(test), "game_id"].nunique()),
        "train_games": int(c.loc[c["case"].isin(train), "game_id"].nunique()),
    }
    out["train"] = {k: evaluate(data, v, train) for k, v in lists.items()}
    out["test"] = {k: evaluate(data, v, test) for k, v in lists.items()}
    for name in ("Ban plan (previous weights)", "Ban plan (tuned on 2025)"):
        out[f"gain_vs_naive: {name}"] = paired_gain(data, lists[name], data["naive"], test)
        out[f"gain_vs_meta: {name}"] = paired_gain(data, lists[name], data["meta"], test)
    return out


def format_report(r: dict) -> str:
    lines = [
        f"Train (2025) cases {r['train_cases']}, test (2026) cases {r['test_cases']}",
        f"Tuned weights: {r['tuned']}  (adopt: {r['adopt_tuned']}; now the defaults: {r['tuned_are_defaults']})",
        f"Games: train {r['train_games']}, test {r['test_games']}; grid {r['grid_size']} settings. Grid top 10:",
        r["grid_top"].round(4).to_string(index=False),
    ]
    for split in ("train", "test"):
        lines.append(f"\n{split.upper()}")
        lines.append(
            "| List | Available | Picked | Pick rate (90% CI) | Game hit rate | We banned | "
            "Win rate when they got one | Their overall | Excess wins |"
        )
        for name, m in r[split].items():
            lines.append(
                f"| {name} | {m['available']} | {m['picked']} | {m['pick_rate']:.1%} "
                f"({m['pick_lo']:.1%}–{m['pick_hi']:.1%}) | {m['game_hit_rate']:.1%} ({m['game_hits']}/"
                f"{m['games']}) | {m['we_banned']:.1%} | {m['win_rate_got']:.1%} ({m['got_wins']}–"
                f"{m['got_games'] - m['got_wins']}; {m['win_lo']:.0%}–{m['win_hi']:.0%}) | "
                f"{m['win_rate_all']:.1%} | {m['excess_got']:+.3f} |"
            )
    for k, v in r.items():
        if k.startswith("gain_"):
            lines.append(f"{k}: {v['diff']:+.3f} (90% CI {v['lo']:+.3f}..{v['hi']:+.3f})")
    return "\n".join(lines)


if __name__ == "__main__":
    import time

    t0 = time.time()
    data = load_cases(rebuild="--rebuild" in sys.argv)
    print(f"cases built/loaded in {time.time() - t0:.0f}s")
    print(format_report(report(data)))
    print(f"({time.time() - t0:.0f}s)")
