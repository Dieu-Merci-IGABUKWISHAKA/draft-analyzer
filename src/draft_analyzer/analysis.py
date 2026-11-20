"""Draft scouting metrics for one team, built on the drafts table.

Every function takes a GRID team_id and returns a DataFrame. Sample sizes
are always included: `games` / `n_games` columns, W-L records, and a
`small_sample` flag where a rate rests on fewer than SMALL_SAMPLE games.

Recency (optional on most functions):
  last_patches=N  keep only the team's N most recent patches
  half_life=H     weight a game by 0.5 ** (patches_ago / H), so a game
                  H patches older than the team's latest counts half.
"patches_ago" counts patches present in the data (some patches have no games).
"""

from __future__ import annotations

import pandas as pd

from draft_analyzer.drafts import load_drafts

SMALL_SAMPLE = 5
PHASE1_BANS = range(1, 7)
PHASE1_PICKS = range(7, 13)
PHASE2_BANS = range(13, 17)
FIRST_PICK_SLOT = 7          # the first-pick team's first pick
SECOND_TEAM_FIRST_SLOT = 8   # the other team's first pick (slot 9 follows at once)
RECENT_PATCHES = 3           # window used for "recent" counts in ban reasons

_drafts_cache: pd.DataFrame | None = None


def _drafts(drafts: pd.DataFrame | None) -> pd.DataFrame:
    global _drafts_cache
    if drafts is not None:
        return drafts
    if _drafts_cache is None:
        _drafts_cache = load_drafts()
    return _drafts_cache


def _patch_key(patch: str) -> tuple[int, ...]:
    return tuple(int(x) for x in str(patch).split("."))


def team_games(team_id: str, drafts: pd.DataFrame | None = None,
               last_patches: int | None = None, half_life: float | None = None,
               vs: str | None = None) -> pd.DataFrame:
    """One row per game the team played: patch, date, won, had_first_pick, weight.

    last_patches and half_life count from the team's latest patch overall;
    vs (an opponent team_id) then keeps only games against that team.
    """
    df = _drafts(drafts)
    patches = sorted(df["patch"].dropna().astype(str).unique(), key=_patch_key)
    patch_idx = {p: i for i, p in enumerate(patches)}

    rows = df[df["team_id"] == team_id]
    games = rows.groupby("game_id", observed=True).agg(
        series_id=("series_id", "first"), date=("date", "first"), patch=("patch", "first"),
        event=("event", "first"), opponent_id=("opponent_id", "first"),
        opponent_name=("opponent_name", "first"), side=("side", "first"),
        winner=("game_winner_team_id", "first"),
    )
    games["won"] = games["winner"] == team_id
    first_pickers = df.loc[df["draft_order"] == FIRST_PICK_SLOT, ["game_id", "team_id"]]
    fp_games = set(first_pickers.loc[first_pickers["team_id"] == team_id, "game_id"])
    games["had_first_pick"] = games.index.isin(fp_games)

    pidx = games["patch"].astype(str).map(patch_idx)
    games["patches_ago"] = pidx.max() - pidx if len(games) else pidx
    if last_patches is not None:
        recent = sorted(pidx.unique())[-last_patches:]
        games = games[pidx.isin(recent)]
    if vs is not None:
        games = games[games["opponent_id"] == vs]
    games["weight"] = 0.5 ** (games["patches_ago"] / half_life) if half_life else 1.0
    return games.drop(columns="winner")


def record(team_id: str, last_patches: int | None = None, vs: str | None = None,
           drafts: pd.DataFrame | None = None) -> dict[str, int]:
    """Games, wins and losses for a team (optionally only against `vs`)."""
    g = team_games(team_id, drafts, last_patches, vs=vs)
    wins = int(g["won"].sum())
    patches = sorted(g["patch"].astype(str).unique(), key=_patch_key)
    return {"games": len(g), "wins": wins, "losses": len(g) - wins,
            "first_patch": patches[0] if patches else None, "last_patch": patches[-1] if patches else None}


def team_options(team_id: str, drafts: pd.DataFrame | None = None, league_prefix: str = "LCS",
                 year: int = 2026) -> pd.DataFrame:
    """Every other team, ordered: teams in `league_prefix` events of `year` first,
    each group sorted by games against team_id, then by name."""
    df = _drafts(drafts)
    games = df.drop_duplicates(["game_id", "team_id"])
    latest = games.sort_values("date").groupby("team_id").agg(name=("team_name", "last"))
    in_league = games[games["event"].astype(str).str.startswith(league_prefix)
                      & (pd.to_datetime(games["date"]).dt.year == year)]["team_id"].unique()
    vs = games[games["opponent_id"] == team_id].groupby("team_id").size()
    out = latest.assign(in_league=latest.index.isin(in_league), games_vs=vs).fillna({"games_vs": 0})
    out["games_vs"] = out["games_vs"].astype(int)
    out = out.drop(index=team_id, errors="ignore")
    return out.sort_values(["in_league", "games_vs", "name"], ascending=[False, False, True])


def window_patches(team_ids: list[str], last_patches: int | None = None,
                   drafts: pd.DataFrame | None = None) -> tuple[str, str] | None:
    """Earliest and latest patch covered by these teams' games in the window."""
    df = _drafts(drafts)
    patches = {str(p) for t in team_ids for p in team_games(t, df, last_patches)["patch"]}
    if not patches:
        return None
    ordered = sorted(patches, key=_patch_key)
    return ordered[0], ordered[-1]


def data_summary(drafts: pd.DataFrame | None = None) -> dict:
    """Patch and date range and size of the drafts table."""
    df = _drafts(drafts)
    patches = sorted(df["patch"].dropna().astype(str).unique(), key=_patch_key)
    return {"first_patch": patches[0], "last_patch": patches[-1], "n_patches": len(patches),
            "first_date": df["date"].min(), "last_date": df["date"].max(),
            "games": df["game_id"].nunique(), "events": sorted(df["event"].astype(str).unique())}


def _record(wins: pd.Series, games: pd.Series) -> pd.Series:
    return wins.astype(int).astype(str) + "-" + (games - wins).astype(int).astype(str)


def champion_pool(team_id: str, last_patches: int | None = None, half_life: float | None = None,
                  drafts: pd.DataFrame | None = None, vs: str | None = None) -> pd.DataFrame:
    """Champions the team picked, per inferred role: games, record, win rate, recency."""
    df = _drafts(drafts)
    games = team_games(team_id, df, last_patches, half_life, vs)
    picks = df[(df["team_id"] == team_id) & (df["action"] == "pick") & df["game_id"].isin(games.index)]
    picks = picks.join(games[["won", "weight"]], on="game_id")

    out = picks.groupby(["champion_name", "role_inferred"], observed=True, dropna=False).agg(
        games=("game_id", "size"), wins=("won", "sum"), weighted_games=("weight", "sum"),
        last_played=("date", "max"), last_patch=("patch", lambda s: max(s.astype(str), key=_patch_key)),
    ).reset_index()
    out = out.rename(columns={"champion_name": "champion", "role_inferred": "role"})
    out["record"] = _record(out["wins"], out["games"])
    out["win_rate"] = out["wins"] / out["games"]
    out["pick_rate"] = out["games"] / len(games)
    out["weighted_pick_rate"] = out["weighted_games"] / games["weight"].sum()
    out["n_team_games"] = len(games)
    out["small_sample"] = out["games"] < SMALL_SAMPLE
    cols = ["champion", "role", "games", "record", "win_rate", "pick_rate", "weighted_pick_rate",
            "last_played", "last_patch", "n_team_games", "small_sample"]
    return out.sort_values(["weighted_games", "games"], ascending=False)[cols].reset_index(drop=True)


def priority_picks(team_id: str, last_patches: int | None = None, half_life: float | None = None,
                   drafts: pd.DataFrame | None = None, vs: str | None = None) -> pd.DataFrame:
    """First-phase picks (slots 7-12) and the team's very first pick, split by
    whether the team had first pick (slot 7) or picked second (slots 8-9 first)."""
    df = _drafts(drafts)
    games = team_games(team_id, df, last_patches, half_life, vs)
    picks = df[(df["team_id"] == team_id) & (df["action"] == "pick")
               & df["draft_order"].isin(PHASE1_PICKS) & df["game_id"].isin(games.index)]
    picks = picks.join(games[["had_first_pick", "won", "weight"]], on="game_id")
    picks["very_first"] = picks["draft_order"] == picks["had_first_pick"].map(
        {True: FIRST_PICK_SLOT, False: SECOND_TEAM_FIRST_SLOT})

    out = picks.groupby(["had_first_pick", "champion_name"], observed=True).agg(
        phase1_picks=("game_id", "size"), very_first_picks=("very_first", "sum"),
        wins=("won", "sum"), weighted=("weight", "sum"),
    ).reset_index()
    n = games.groupby("had_first_pick").agg(n_games=("won", "size"), w_games=("weight", "sum"))
    out = out.join(n, on="had_first_pick")
    out["draft_position"] = out["had_first_pick"].map({True: "first pick", False: "second pick"})
    out["phase1_rate"] = out["phase1_picks"] / out["n_games"]
    out["very_first_rate"] = out["very_first_picks"] / out["n_games"]
    out["weighted_phase1_rate"] = out["weighted"] / out["w_games"]
    out["record"] = _record(out["wins"], out["phase1_picks"])
    out["small_sample"] = out["phase1_picks"] < SMALL_SAMPLE
    out = out.rename(columns={"champion_name": "champion"})
    cols = ["draft_position", "champion", "very_first_picks", "phase1_picks", "n_games", "very_first_rate",
            "phase1_rate", "weighted_phase1_rate", "record", "small_sample"]
    return out.sort_values(["draft_position", "weighted", "very_first_picks"],
                           ascending=[True, False, False])[cols].reset_index(drop=True)


def _bans(team_id: str, by_team: bool, last_patches, half_life, drafts, vs=None) -> pd.DataFrame:
    df = _drafts(drafts)
    games = team_games(team_id, df, last_patches, half_life, vs)
    who = "team_id" if by_team else "opponent_id"
    bans = df[(df[who] == team_id) & (df["action"] == "ban") & df["game_id"].isin(games.index)]
    bans = bans.join(games[["weight"]], on="game_id")
    bans["phase"] = bans["draft_order"].map(lambda s: 1 if s in PHASE1_BANS else 2)

    out = bans.groupby("champion_name", observed=True).agg(
        bans=("game_id", "size"),
        phase1_bans=("phase", lambda s: (s == 1).sum()),
        phase2_bans=("phase", lambda s: (s == 2).sum()),
        weighted=("weight", "sum"),
    ).reset_index().rename(columns={"champion_name": "champion"})
    out["n_games"] = len(games)
    out["ban_rate"] = out["bans"] / len(games)
    out["weighted_ban_rate"] = out["weighted"] / games["weight"].sum()
    out["small_sample"] = len(games) < SMALL_SAMPLE
    cols = ["champion", "bans", "phase1_bans", "phase2_bans", "n_games", "ban_rate", "weighted_ban_rate", "small_sample"]
    return out.sort_values(["weighted", "bans"], ascending=False)[cols].reset_index(drop=True)


def bans_by(team_id: str, last_patches: int | None = None, half_life: float | None = None,
            drafts: pd.DataFrame | None = None, vs: str | None = None) -> pd.DataFrame:
    """Champions this team bans (only vs `vs` if given). Phase 1 = slots 1-6, phase 2 = slots 13-16."""
    return _bans(team_id, True, last_patches, half_life, drafts, vs)


def bans_against(team_id: str, last_patches: int | None = None, half_life: float | None = None,
                 drafts: pd.DataFrame | None = None, vs: str | None = None) -> pd.DataFrame:
    """Champions opponents (only `vs` if given) ban against this team."""
    return _bans(team_id, False, last_patches, half_life, drafts, vs)


# Tunable knobs for recommend_bans (meant to become dashboard sliders).
DEFAULT_BAN_WEIGHTS: dict[str, float] = {
    "win_rate": 3.0,          # x pick_rate x win-rate edge over wr_pivot (signed: below the pivot it subtracts)
    "pick": 0.0,              # x pick_rate, regardless of results (raise to value sheer presence)
    "banned_by_others": 0.25, # x rate other teams ban it against this opponent
    "our_pick_penalty": 0.5,  # positive scores x (1 - this x our own pick rate)
    "wr_pivot": 0.5,          # win rate at which the win-rate term is zero
    "prior_games": 5.0,       # smoothing: imaginary games at the opponent's overall win rate
    "half_life": 3.0,         # patches until a game counts half
}


def recommend_bans(our_team_id: str, opponent_id: str, n: int = 5, weights: dict[str, float] | None = None,
                   drafts: pd.DataFrame | None = None, last_patches: int | None = None) -> pd.DataFrame:
    """Rank champions to ban against an opponent.

    edge  = (smoothed_wr - wr_pivot) / wr_pivot          +ve if they win on it, -ve if they lose
    score = weighted_pick_rate * (pick + win_rate * edge)
          + banned_by_others * weighted_banned_against_rate
    positive scores x (1 - our_pick_penalty * our_pick_rate)

    With the default pick=0, a champion the opponent wins under 50% on
    gets a negative pick term: banning it would take away a champion they
    do badly with. smoothed_wr adds `prior_games` imaginary games at the
    opponent's overall win rate, so a 1-0 record isn't treated as 100%.
    `weights` overrides any subset of DEFAULT_BAN_WEIGHTS. last_patches limits
    both teams to their N most recent patches (on top of the half-life decay).
    """
    unknown = set(weights or {}) - set(DEFAULT_BAN_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown ban weights: {sorted(unknown)}")
    w = {**DEFAULT_BAN_WEIGHTS, **(weights or {})}
    half_life, prior_games = w["half_life"], w["prior_games"]
    df = _drafts(drafts)
    games = team_games(opponent_id, df, last_patches, half_life)
    cols = ["champion", "score", "picks", "wins", "record", "recent_picks", "smoothed_wr", "wr_edge",
            "weighted_pick_rate", "banned_against", "our_picks", "n_our_games", "n_opp_games",
            "reason_lines", "reason"]
    if games.empty:
        return pd.DataFrame(columns=cols)
    W = games["weight"].sum()
    opp_wr = games["won"].mean()
    recent_ids = set(games.index[games["patch"].astype(str).isin(
        sorted(games["patch"].astype(str).unique(), key=_patch_key)[-RECENT_PATCHES:])])

    in_games = df[df["game_id"].isin(games.index)].join(games[["won", "weight"]], on="game_id")
    picks = in_games[(in_games["team_id"] == opponent_id) & (in_games["action"] == "pick")]
    banned_vs = in_games[(in_games["opponent_id"] == opponent_id) & (in_games["action"] == "ban")]

    p = picks.groupby("champion_id").agg(
        champion=("champion_name", "first"), picks=("game_id", "size"), wins=("won", "sum"),
        w_picks=("weight", "sum"), recent_picks=("game_id", lambda s: s.isin(recent_ids).sum()),
    )
    b = banned_vs.groupby("champion_id").agg(
        champion_b=("champion_name", "first"), banned_against=("game_id", "size"), w_banned=("weight", "sum"))
    out = p.join(b, how="outer")
    out["champion"] = out["champion"].fillna(out["champion_b"])
    out = out.drop(columns="champion_b").fillna({"picks": 0, "wins": 0, "w_picks": 0, "recent_picks": 0,
                                                 "banned_against": 0, "w_banned": 0})

    ours = team_games(our_team_id, df, last_patches, half_life)
    our_picks = df[(df["team_id"] == our_team_id) & (df["action"] == "pick") & df["game_id"].isin(ours.index)]
    our_picks = our_picks.join(ours[["weight"]], on="game_id").groupby("champion_id").agg(
        our_picks=("game_id", "size"), our_w=("weight", "sum"))
    out = out.join(our_picks).fillna({"our_picks": 0, "our_w": 0})

    out["weighted_pick_rate"] = out["w_picks"] / W
    out["smoothed_wr"] = (out["wins"] + prior_games * opp_wr) / (out["picks"] + prior_games)
    out["weighted_banned_against_rate"] = out["w_banned"] / W
    out["our_pick_rate"] = out["our_w"] / ours["weight"].sum() if len(ours) else 0.0
    out["wr_edge"] = (out["smoothed_wr"] - w["wr_pivot"]) / w["wr_pivot"]
    score = (out["weighted_pick_rate"] * (w["pick"] + w["win_rate"] * out["wr_edge"])
             + w["banned_by_others"] * out["weighted_banned_against_rate"])
    out["score"] = score.where(score <= 0, score * (1 - w["our_pick_penalty"] * out["our_pick_rate"]))

    n_games, n_recent = len(games), len(recent_ids)

    def reason_lines(r) -> list[str]:
        """Short plain-English lines; the last (optional) one is about our own use of it."""
        lines = [f"Picked {int(r.picks)} of {n_games} games · {int(r.recent_picks)} in last {RECENT_PATCHES} patches"]
        if r.picks:
            wr = f"Win rate {r.wins / r.picks:.0%} ({int(r.wins)}–{int(r.picks - r.wins)})"
            lines.append(wr + (" · small sample" if r.picks < SMALL_SAMPLE else ""))
        if r.banned_against:
            lines.append(f"Banned vs them in {int(r.banned_against)} games")
        if r.our_picks:
            lines.append(f"We pick it too ({int(r.our_picks)} of our {len(ours)} games)")
        return lines

    out = out.sort_values("score", ascending=False).head(n).reset_index(drop=True)
    out["reason_lines"] = out.apply(reason_lines, axis=1)
    out["reason"] = out["reason_lines"].str.join(" · ")
    out["record"] = _record(out["wins"], out["picks"])
    out["n_opp_games"] = n_games
    out["n_our_games"] = len(ours)
    return out[cols]
