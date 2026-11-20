"""Early-game macro context from the event feeds, linked to the draft.

Per game: who took first blood, the first dragon, the first void grub, the first rift
herald and the first tower, and at what game time (the game clock in the event's own
snapshot). Events are collected in the same pass as the drafts table (drafts.scan_series)
and cached in data_cache/objective_events.parquet.

Event names, checked in the feeds:
  first blood   player-killed-player (executions and multikill events left out)
  first dragon  player-completed-slayDragon, or the drake-type events (slayMountainDrake,
                ...): early-2025 feeds have only the drake-type ones
  first grubs   player-completed-slayVoidGrub (the team that killed the first grub)
  first herald  player-completed-slayRiftHerald
  first tower   player-destroyed-tower / team-destroyed-tower (minions). Credited to the side
                opposite the tower's owner (its id starts with "blue-"/"red-"): in 8 of 20,647
                events the credited team was on the tower's own side.

Patterns (per team) and whether they hold up in later games (`replication`, the same check
as insights.py, split dates July 2025 and April 2026):
  objective rate    how often they take the objective first (vs 50%)
  objective timing  how much earlier/later than the league they take it, when they do
  jungler link      their rate with one jungler champion vs with their other junglers
Only kinds of pattern that held up are shown in the app (insights.MIN_RELIABILITY), and only
rows that pass insights.MIN_CONFIDENCE.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer import insights as I  # noqa: E402
from draft_analyzer.config import CACHE_DIR  # noqa: E402
from draft_analyzer.drafts import DATA_DIR, LOL_DIR, OBJECTIVES_PATH, scan_series  # noqa: E402

OBJECTIVES = {  # key -> (label, predicate on the event type)
    "first_blood": ("First blood", lambda t: t == "player-killed-player"),
    "first_dragon": ("First dragon", lambda t: t.endswith("slayDragon") or t.endswith("Drake")),
    "first_grubs": ("First void grub", lambda t: t.endswith("slayVoidGrub")),
    "first_herald": ("First rift herald", lambda t: t.endswith("slayRiftHerald")),
    "first_tower": ("First tower", lambda t: t.endswith("destroyed-tower")),
}
MIN_RATE_GAMES = 15  # games for an objective-rate pattern
MIN_TIMING_GAMES = 8  # objectives taken, for a timing pattern
MIN_JUNGLER_GAMES = 5  # games with the jungler (and without) for a jungler link
LATE_MIN = 3  # later periods are short
SPLITS = I.REPLICATION_SPLITS


# --- per-game firsts ----------------------------------------------------------------------
def load_objective_events(rebuild: bool = False) -> pd.DataFrame:
    """The cached objective events; streams the feeds (about 40 s) if the cache is missing."""
    if rebuild or not OBJECTIVES_PATH.exists():
        series = pd.read_csv(LOL_DIR / "series.csv", dtype=str, keep_default_na=False)
        rows = []
        for s in series.itertuples(index=False):
            _, ev = scan_series(DATA_DIR / s.file)
            rows += [{**e, "series_id": s.series_id} for e in ev]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(OBJECTIVES_PATH, index=False)
    return pd.read_parquet(OBJECTIVES_PATH)


def apply_undos(events: pd.DataFrame) -> pd.DataFrame:
    """Drop each undo event ("!") and the latest earlier matching event in the same game."""
    if not events["undo"].any():
        return events[~events["undo"]]
    keep = np.ones(len(events), dtype=bool)
    ev = events.reset_index(drop=True)
    for i in np.flatnonzero(ev["undo"].to_numpy()):
        keep[i] = False
        same = (
            (ev["game_id"] == ev.at[i, "game_id"])
            & (ev["type"] == ev.at[i, "type"])
            & (ev["team_id"] == ev.at[i, "team_id"])
            & (ev["target_id"] == ev.at[i, "target_id"])
            & ~ev["undo"]
            & (ev.index < i)
            & keep
        )
        if same.any():
            keep[same[same].index[-1]] = False
    return ev[keep]


def first_objectives(events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per game (index game_id): for each objective, the team that took it first
    (`<key>_team`) and the game time in seconds (`<key>_s`); NaN if it never happened."""
    ev = apply_undos(events)
    ev = ev[ev["game_id"].isin(games.index)].copy()
    g = games.reindex(ev["game_id"])
    tower = ev["type"].str.endswith("destroyed-tower").to_numpy()
    owner = ev["target_id"].fillna("").str.split("-").str[0].to_numpy()
    ev["team_id"] = np.where(
        tower & (owner == "red"),
        g["blue_team"].to_numpy(),
        np.where(tower & (owner == "blue"), g["red_team"].to_numpy(), ev["team_id"].to_numpy()),
    )
    ok = (ev["team_id"].to_numpy() == g["blue_team"].to_numpy()) | (
        ev["team_id"].to_numpy() == g["red_team"].to_numpy()
    )
    ev = ev[ok].sort_values(["game_id", "seconds", "order"])
    out = pd.DataFrame(index=games.index)
    for key, (_, is_kind) in OBJECTIVES.items():
        first = ev[ev["type"].map(is_kind)].drop_duplicates("game_id").set_index("game_id")
        out[f"{key}_team"] = first["team_id"].reindex(out.index)
        out[f"{key}_s"] = first["seconds"].reindex(out.index).astype(float)
    return out


def mmss(seconds: float) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "–"
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"


# --- per-team view ------------------------------------------------------------------------
class Macro:
    """Macro profiles for any team. Built on an insights.Scout (shared tables)."""

    def __init__(self, scout: I.Scout, events: pd.DataFrame | None = None):
        self.scout = scout
        self.firsts = first_objectives(load_objective_events() if events is None else events, scout.games)
        a = scout.actions
        jg = a[(a["action"] == "pick") & (a["role"] == "jungle")]
        self.jungler = jg.drop_duplicates(["game_id", "team_id"]).set_index(["game_id", "team_id"])["champion_id"]

    def team_table(self, team: str, tg: pd.DataFrame) -> pd.DataFrame:
        """The team's games with, per objective, took (1/0, NaN if nobody took it) and seconds."""
        f = self.firsts.reindex(tg.index)
        out = pd.DataFrame(index=tg.index)
        for key in OBJECTIVES:
            happened = f[f"{key}_team"].notna()
            out[key] = np.where(happened, (f[f"{key}_team"] == team).astype(float), np.nan)
            out[f"{key}_s"] = f[f"{key}_s"]
        out["jungler"] = self.jungler.reindex(pd.MultiIndex.from_arrays([tg.index, np.full(len(tg), team)])).to_numpy()
        out["date"] = tg["date"]
        return out

    def league_medians(self, tg: pd.DataFrame) -> dict[str, float]:
        """Median time of each first objective over all games in the team's date range."""
        g = self.scout.games
        span = self.firsts[(g["date"] >= tg["date"].min()) & (g["date"] <= tg["date"].max())]
        return {k: float(span[f"{k}_s"].median()) for k in OBJECTIVES}

    def profile(self, team: str, tg: pd.DataFrame) -> pd.DataFrame:
        """Per objective: games where it happened, how often the team took it first (with a
        Wilson 90% range), their median time when they took it, and the league median."""
        t, league = self.team_table(team, tg), self.league_medians(tg)
        rows = []
        for key, (label, _) in OBJECTIVES.items():
            x = t[key].dropna()
            k, n = int(x.sum()), len(x)
            lo, hi = I.wilson(k, n)
            took = t.loc[t[key] == 1, f"{key}_s"]
            rows.append(
                {
                    "key": key,
                    "objective": label,
                    "games": n,
                    "took": k,
                    "rate": k / n if n else np.nan,
                    "lo": lo,
                    "hi": hi,
                    "their_median_s": float(took.median()) if len(took) else np.nan,
                    "taken_n": len(took),
                    "league_median_s": league[key],
                    "rate_conf": I.one_prop(k, n, 0.5),
                    "timing_conf": _timing_conf(took, league[key]),
                }
            )
        return pd.DataFrame(rows)

    def jungler_links(self, team: str, tg: pd.DataFrame, min_games: int = MIN_JUNGLER_GAMES) -> pd.DataFrame:
        """Per (jungler champion, objective): the team's rate with that jungler vs with
        their other junglers, with confidence (two-proportion test)."""
        t = self.team_table(team, tg)
        rows = []
        for jg, n_jg in t["jungler"].value_counts().items():
            if n_jg < min_games:
                continue
            with_j = t["jungler"] == jg
            for key, (label, _) in OBJECTIVES.items():
                a, b = t.loc[with_j, key].dropna(), t.loc[~with_j, key].dropna()
                if min(len(a), len(b)) < min_games:
                    continue
                diff, conf = I.two_prop(int(a.sum()), len(a), int(b.sum()), len(b))
                lo, hi = I.wilson(int(a.sum()), len(a))
                rows.append(
                    {
                        "jungler_id": jg,
                        "jungler": self.scout.champ_names.get(jg, jg),
                        "key": key,
                        "objective": label,
                        "n": len(a),
                        "took": int(a.sum()),
                        "rate": a.mean(),
                        "lo": lo,
                        "hi": hi,
                        "n_other": len(b),
                        "rate_other": b.mean(),
                        "diff": diff,
                        "confidence": conf,
                    }
                )
        return pd.DataFrame(rows)

    # --- summary for the app ----------------------------------------------------------------
    def summary(
        self,
        team: str,
        last_patches: int | None = I.DEFAULT_PATCHES,
        replication: pd.DataFrame | None = None,
        top_links: int = 3,
    ) -> dict:
        """What the Opponent scouting tab shows: objective rows and jungler lines, each only
        if its kind of pattern held up in later games and the row itself is confident
        (insights.MIN_CONFIDENCE). `held` says which kinds held up."""
        tg = self.scout.team_games(team, last_patches)
        prof, links = self.profile(team, tg), self.jungler_links(team, tg)
        held = _held(replication)
        prof["show_rate"] = prof["key"].map(lambda k: held.get(("objective rate", k), False)) & (
            prof["rate_conf"] >= I.MIN_CONFIDENCE
        )
        prof["show_timing"] = prof["key"].map(lambda k: held.get(("objective timing", k), False)) & (
            prof["timing_conf"] >= I.MIN_CONFIDENCE
        )
        if len(links):
            links["held"] = links["key"].map(lambda k: held.get(("jungler link", k), False))
            shown = links[links["held"] & (links["confidence"] >= I.MIN_CONFIDENCE)]
            shown = shown.sort_values("confidence", ascending=False).head(top_links)
        else:
            shown = links
        name = self.scout.team_names.get(team, team)
        lines = [
            f"With {r.jungler}, {name} take the {r.objective.lower()} {r.rate:.0%} of the time "
            f"({r.took} of {r.n} games; 90% range {r.lo:.0%}–{r.hi:.0%}) vs {r.rate_other:.0%} with other "
            f"junglers ({r.n_other} games)."
            for r in shown.itertuples()
        ]
        return {"team": name, "games": len(tg), "profile": prof, "jungler_lines": lines, "links": links, "held": held}


def _timing_conf(times: pd.Series, league_median: float) -> float:
    """Sign test: confidence that their times sit on one side of the league median."""
    t = times.dropna()
    t = t[t != league_median]
    return I.one_prop(int((t < league_median).sum()), len(t), 0.5) if len(t) else 0.0


def _held(replication: pd.DataFrame | None) -> dict:
    """(kind, objective) -> True if that kind of pattern held up in later games
    (insights.holds_up: kept >= 25% and the later gap's 90% interval above zero)."""
    if replication is None or replication.empty:
        return {}
    return {(r.kind, r.key): I.holds_up(r.survives, r.late_lo) for r in replication.itertuples()}


# --- do macro patterns hold up in later games? --------------------------------------------
def replication(macro: Macro, splits: Sequence[str] = SPLITS) -> pd.DataFrame:
    """For every team and split date, measure each pattern on games before the split and
    again on games after it. One row per checkable pattern: kind, objective, early effect
    and confidence, later effect signed so > 0 means the same direction."""
    rows = []
    sc = macro.scout
    for split in splits:
        for team in sc.team_names.index:
            early, late = sc.team_games(team, None, before=split), sc.team_games(team, None, since=split)
            if len(early) < MIN_RATE_GAMES or len(late) < 2 * LATE_MIN:
                continue
            pe, pl = macro.profile(team, early), macro.profile(team, late)
            for (_, e), (_, lt) in zip(pe.iterrows(), pl.iterrows(), strict=True):
                if e["games"] >= MIN_RATE_GAMES and lt["games"] >= LATE_MIN:
                    sign = np.sign(e["rate"] - 0.5) or 1
                    rows.append(
                        {
                            "split": split,
                            "team": team,
                            "kind": "objective rate",
                            "key": e["key"],
                            "early_effect": abs(e["rate"] - 0.5),
                            "early_conf": e["rate_conf"],
                            "late_effect": sign * (lt["rate"] - 0.5),
                        }
                    )
                if e["taken_n"] >= MIN_TIMING_GAMES and lt["taken_n"] >= LATE_MIN:
                    d_e = e["their_median_s"] - e["league_median_s"]
                    d_l = lt["their_median_s"] - lt["league_median_s"]
                    sign = np.sign(d_e) or 1
                    rows.append(
                        {
                            "split": split,
                            "team": team,
                            "kind": "objective timing",
                            "key": e["key"],
                            "early_effect": abs(d_e),
                            "early_conf": e["timing_conf"],
                            "late_effect": sign * d_l,
                        }
                    )
            le = macro.jungler_links(team, early)
            ll = macro.jungler_links(team, late, min_games=LATE_MIN)
            if len(le) and len(ll):
                m = le.merge(ll, on=["jungler_id", "key"], suffixes=("", "_late"))
                for r in m.itertuples():
                    sign = np.sign(r.diff) or 1
                    rows.append(
                        {
                            "split": split,
                            "team": team,
                            "kind": "jungler link",
                            "key": r.key,
                            "early_effect": abs(r.diff),
                            "early_conf": r.confidence,
                            "late_effect": sign * r.diff_late,
                        }
                    )
    return pd.DataFrame(rows)


def replication_summary(rep: pd.DataFrame, min_confidence: float = I.MIN_CONFIDENCE) -> pd.DataFrame:
    """Per (kind, objective), over patterns that pass the confidence bar: checks, same
    direction later, average effect before/after, and the share kept (as insights.py).
    Timing effects are in seconds; rate effects in share points."""
    out = []
    for (kind, key), g_all in rep.groupby(["kind", "key"]):
        g = g_all[g_all["early_conf"] >= min_confidence]
        if g.empty:
            out.append(
                {
                    "kind": kind,
                    "key": key,
                    "objective": OBJECTIVES[key][0],
                    "checks": 0,
                    "same_direction": np.nan,
                    "early_effect": np.nan,
                    "late_effect": np.nan,
                    "late_lo": np.nan,
                    "late_hi": np.nan,
                    "all_checks": len(g_all),
                    "all_same_direction": float((g_all["late_effect"] > 0).mean()),
                    "survives": 0.0,
                }
            )
            continue
        late = float(g["late_effect"].mean())
        half = I.Z90 * g["late_effect"].std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else np.nan
        out.append(
            {
                "kind": kind,
                "key": key,
                "objective": OBJECTIVES[key][0],
                "checks": len(g),
                "same_direction": float((g["late_effect"] > 0).mean()),
                "early_effect": float(g["early_effect"].mean()),
                "late_effect": late,
                "late_lo": late - half,
                "late_hi": late + half,
                "all_checks": len(g_all),
                "all_same_direction": float((g_all["late_effect"] > 0).mean()),
                "survives": float(np.clip(late / g["early_effect"].mean(), 0, 1)),
            }
        )
    return pd.DataFrame(out).sort_values(["kind", "survives"], ascending=[True, False]).reset_index(drop=True)


def load_replication(macro: Macro, rebuild: bool = False) -> pd.DataFrame:
    """replication_summary, cached in data_cache/ (aggregate numbers only)."""
    import joblib

    path = CACHE_DIR / "macro_replication.joblib"
    key = (
        str(macro.scout.games["date"].max().date()),
        SPLITS,
        MIN_RATE_GAMES,
        MIN_TIMING_GAMES,
        MIN_JUNGLER_GAMES,
        LATE_MIN,
        I.MIN_CONFIDENCE,
    )
    if not rebuild and path.exists():
        saved = joblib.load(path)
        if saved.get("key") == key:
            return saved["summary"]
    summary = replication_summary(replication(macro))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "summary": summary}, path)
    return summary


if __name__ == "__main__":
    from draft_analyzer import model as M
    from draft_analyzer.drafts import load_drafts

    pd.set_option("display.width", 220)
    mac = Macro(I.Scout(load_drafts()))
    f = mac.firsts
    print("coverage:", {k: f"{f[f'{k}_team'].notna().mean():.1%}" for k in OBJECTIVES})
    rep = load_replication(mac, rebuild=True)
    print(rep.round(3).to_string(index=False))
    for name in ("LYON", "FlyQuest", "Cloud9 Kia"):
        s = mac.summary(M._resolve_team(name, mac.scout.drafts), replication=rep)
        p = s["profile"]
        print(f"\n== {name} ({s['games']} games)")
        print(
            p[
                [
                    "objective",
                    "games",
                    "took",
                    "rate",
                    "lo",
                    "hi",
                    "their_median_s",
                    "league_median_s",
                    "rate_conf",
                    "timing_conf",
                    "show_rate",
                    "show_timing",
                ]
            ]
            .round(2)
            .to_string(index=False)
        )
        for line in s["jungler_lines"]:
            print("  ", line)
