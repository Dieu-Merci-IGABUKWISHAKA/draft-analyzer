"""Draft win-probability model: P(blue side wins) from team strength, side and the draft.

One row per game, seen from BLUE side. Every feature for a game is computed
only from games played on EARLIER DATES (strictly before the game's date),
so nothing from the game itself, other games that day, or the future leaks in.

Features ("blue minus red" for the d_ columns):
  elo_diff         Elo rating difference, updated game by game (K tuned on early data)
  blue_first_pick  1 if blue makes the first pick. Side itself is the model's
                   intercept; since 2026 ("first selection") first pick can be on red,
                   so side and first pick are separate.
  d_champ_wr       meta strength: champion's smoothed win rate - 50%, all teams, last 3 patches
  d_presence       share of games the champion was picked or banned in that window
  d_priority       how often the champion is taken in the first pick phase (slots 7-12),
                   smoothed, minus the 60% a random pick would have
  d_team_games / d_team_edge      the team's games, and smoothed win rate vs its own
                   average, on the champion in that role (last 365 days)
  d_player_games / d_player_edge  the same for the player who picked it
  d_denied         per ban: log(1 + games the opponent played on that champion)
  d_n_picks, d_n_bans             draft progress (partial drafts)
  matchup          sum over roles where both sides have picked: smoothed prior win rate
                   of blue's champion vs red's in that role, minus 50% (0 with < 3 games)

Draft features are sums over the picks made so far, so a pick that hasn't been
made adds nothing; the model is trained on every stage of every draft (after 0..20
actions, each stage weighted 1/21) so it is calibrated mid-draft too.

Validation is walk-forward by calendar month: for each month, train on every
earlier game, test on that month. Model settings (C) and Platt calibration for a
month are fitted only on earlier months' out-of-sample predictions.

Usage:
    python src/draft_analyzer/model.py      # walk-forward report, then save the final model
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer.analysis import FIRST_PICK_SLOT, PHASE1_PICKS, _patch_key  # noqa: E402
from draft_analyzer.config import CACHE_DIR  # noqa: E402
from draft_analyzer.drafts import load_drafts  # noqa: E402

MODEL_PATH = CACHE_DIR / "draft_model.joblib"

WARMUP_START = "2025-04-01"   # first walk-forward month: feeds later months' C choice and calibration
REPORT_START = "2025-07-01"   # months from here on are reported
ELO_BURN_IN_DAYS = 60
ELO_K_GRID = [8, 12, 16, 20, 24, 32, 40, 48, 64, 80]
C_GRID = [0.003, 0.01, 0.03, 0.1, 1.0]
DEFAULT_C = 0.01              # first fold only (no earlier folds to choose with)
MIN_CALIBRATION_GAMES = 100
N_BOOT = 2000

ROLES = ["top", "jungle", "mid", "bot", "support"]
STAGES = list(range(21))      # draft actions completed
FULL = 20
PHASE1_BASE = 0.6             # 6 of the 10 picks in a draft are in phase 1

PICK_COLS = ["champ_wr", "presence", "priority", "team_games", "team_edge", "player_games", "player_edge"]
DRAFT_COLS = PICK_COLS + ["denied", "n_picks", "n_bans"]
DRAFT_FEATURES = [f"d_{c}" for c in DRAFT_COLS] + ["matchup"]
ELO_FEATURES = ["elo_diff"]
SIDE_FEATURES = ["elo_diff", "blue_first_pick"]
FEATURES = SIDE_FEATURES + DRAFT_FEATURES

FEATURE_LABELS = {
    "elo_diff": "Team strength (Elo)",
    "blue_first_pick": "Blue has first pick",
    "d_champ_wr": "Champion meta win rate",
    "d_presence": "Champion pick/ban presence",
    "d_priority": "Champion first-phase priority",
    "d_team_games": "Team games on champion (role)",
    "d_team_edge": "Team results on champion (role)",
    "d_player_games": "Player games on champion",
    "d_player_edge": "Player results on champion",
    "d_denied": "Bans hitting opponent's pool",
    "d_n_picks": "Picks made so far",
    "d_n_bans": "Bans made so far",
    "matchup": "Lane matchups",
}


@dataclass(frozen=True)
class FeatureConfig:
    """Fixed in advance (not tuned), except elo_k, which is tuned on games before REPORT_START."""
    elo_k: float = 32.0
    elo_init: float = 1500.0
    champ_patches: int = 3        # meta window: last N patches with games before the date
    champ_prior: float = 20.0     # imaginary games at 50% per champion
    priority_prior: float = 5.0   # imaginary picks at PHASE1_BASE
    comfort_days: int = 365       # team/player comfort and matchups: last N days
    team_prior: float = 5.0       # imaginary games at the team's own win rate
    player_prior: float = 5.0     # imaginary games at the player's own win rate
    matchup_prior: float = 10.0   # imaginary games at 50% per matchup
    matchup_min_games: int = 3    # fewer prior meetings than this -> 50%


# --- tables -------------------------------------------------------------------------------
def prepare(drafts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, actions). games: one row per game (index game_id), sorted by date/series/
    game_number, with blue/red teams, blue_first_pick, blue_won and league. actions: one
    row per pick/ban with plain str columns, side, role, player_id and `won` for its team."""
    d = drafts.copy()
    if "player_id" not in d:
        d["player_id"] = None
    for col in ("patch", "side", "action", "role_inferred", "event", "player_id"):
        d[col] = d[col].astype(object).where(d[col].notna(), None)
        d[col] = d[col].map(lambda v: None if v is None else str(v))
    d["date"] = pd.to_datetime(d["date"])

    g = d.drop_duplicates("game_id").set_index("game_id")[
        ["series_id", "game_number", "event", "date", "patch", "game_winner_team_id"]]
    sides = d.drop_duplicates(["game_id", "team_id"])
    for side in ("blue", "red"):
        g[f"{side}_team"] = sides[sides["side"] == side].set_index("game_id")["team_id"]
    g = g.dropna(subset=["blue_team", "red_team", "game_winner_team_id"])
    fp = d[(d["action"] == "pick") & (d["draft_order"] == FIRST_PICK_SLOT)].set_index("game_id")["team_id"]
    g["blue_first_pick"] = (fp.reindex(g.index) == g["blue_team"]).astype(int)
    g["blue_won"] = (g["game_winner_team_id"] == g["blue_team"]).astype(int)
    g["league"] = g["event"].str.split().str[0]
    g = g.sort_values(["date", "series_id", "game_number"])

    a = d[d["game_id"].isin(g.index)][["game_id", "date", "patch", "team_id", "opponent_id", "side", "action",
                                       "draft_order", "champion_id", "role_inferred", "player_id"]].copy()
    a = a.rename(columns={"role_inferred": "role"})
    a["role"] = a["role"].fillna("?")
    a["won"] = (a["team_id"] == a["game_id"].map(g["game_winner_team_id"])).astype(int)
    return g, a.reset_index(drop=True)


# --- Elo ----------------------------------------------------------------------------------
def elo(games: pd.DataFrame, k: float, init: float = 1500.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """(blue, red) ratings before each game, frozen for a whole date (games on one date
    all see the ratings from the end of the previous date), plus the final ratings."""
    ratings: dict[str, float] = {}
    dates, blue, red, won = (games["date"].to_numpy(), games["blue_team"].to_numpy(),
                             games["red_team"].to_numpy(), games["blue_won"].to_numpy())
    pre_b, pre_r = np.empty(len(games)), np.empty(len(games))
    i, n = 0, len(games)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            pre_b[j], pre_r[j] = ratings.get(blue[j], init), ratings.get(red[j], init)
            j += 1
        for t in range(i, j):
            ra, rb = ratings.get(blue[t], init), ratings.get(red[t], init)
            delta = k * (won[t] - 1 / (1 + 10 ** ((rb - ra) / 400)))
            ratings[blue[t]], ratings[red[t]] = ra + delta, rb - delta
        i = j
    return pre_b, pre_r, ratings


def elo_prob(diff) -> np.ndarray:
    return 1 / (1 + 10 ** (-np.asarray(diff, dtype=float) / 400))


def tune_elo_k(games: pd.DataFrame, before: str = REPORT_START, grid=ELO_K_GRID,
               init: float = 1500.0) -> tuple[float, pd.DataFrame]:
    """K with the best one-step-ahead log loss of raw Elo on games before `before`."""
    g = games[games["date"] < pd.Timestamp(before)]
    scored = (g["date"] >= g["date"].min() + pd.Timedelta(days=ELO_BURN_IN_DAYS)).to_numpy()
    rows = []
    for k in grid:
        pre_b, pre_r, _ = elo(g, k, init)
        rows.append({"k": k, "log_loss": log_loss(g["blue_won"].to_numpy()[scored],
                                                  elo_prob(pre_b - pre_r)[scored]),
                     "games": int(scored.sum())})
    table = pd.DataFrame(rows)
    return float(table.loc[table["log_loss"].idxmin(), "k"]), table


# --- history (everything strictly before a date) -----------------------------------------
@dataclass
class History:
    champ: pd.DataFrame        # champion_id -> champ_wr, presence, priority, picks, wins, bans, window_games
    champ_roles: pd.Series     # (champion_id, role) -> games, for guessing roles live
    team: pd.DataFrame         # (team_id, champion_id, role) -> games, wins, team_games, team_edge
    team_all: pd.Series        # (team_id, champion_id) -> games in any role (for bans)
    player: pd.DataFrame       # (player_id, champion_id) -> games, wins, player_games, player_edge
    matchup: pd.DataFrame      # (role, champion_id, opp_champion_id) -> games, wins, edge
    window: list[str]


def history_before(games: pd.DataFrame, actions: pd.DataFrame, before, cfg: FeatureConfig,
                   patch: str | None = None) -> History:
    before = pd.Timestamp(before)
    prior_g = games[games["date"] < before]
    prior_a = actions[actions["date"] < before]

    # Meta: all teams, last N patches (ending at `patch` if given).
    patches = sorted(prior_g["patch"].dropna().unique(), key=_patch_key)
    if patch is not None:
        patches = [p for p in patches if _patch_key(p) <= _patch_key(patch)]
    window = patches[-cfg.champ_patches:]
    n_w = int(prior_g["patch"].isin(window).sum())
    wa = prior_a[prior_a["patch"].isin(window)]
    wp = wa[wa["action"] == "pick"].assign(phase1=lambda x: x["draft_order"].isin(PHASE1_PICKS))
    champ = wp.groupby("champion_id").agg(picks=("won", "size"), wins=("won", "sum"), phase1=("phase1", "sum"))
    champ = champ.join(wa[wa["action"] == "ban"].groupby("champion_id").size().rename("bans"), how="outer").fillna(0)
    champ["champ_wr"] = (champ["wins"] + cfg.champ_prior * 0.5) / (champ["picks"] + cfg.champ_prior) - 0.5
    champ["presence"] = (champ["picks"] + champ["bans"]) / n_w if n_w else 0.0
    champ["priority"] = ((champ["phase1"] + cfg.priority_prior * PHASE1_BASE)
                         / (champ["picks"] + cfg.priority_prior) - PHASE1_BASE)
    champ["window_games"] = n_w

    # Comfort and matchups: picks in the last `comfort_days` days.
    recent = prior_a[(prior_a["action"] == "pick")
                     & (prior_a["date"] >= before - pd.Timedelta(days=cfg.comfort_days))]
    tg = recent.drop_duplicates(["game_id", "team_id"]).groupby("team_id")["won"].agg(["size", "sum"])
    team_wr = (tg["sum"] + cfg.team_prior * 0.5) / (tg["size"] + cfg.team_prior)
    team = recent.groupby(["team_id", "champion_id", "role"])["won"].agg(games="size", wins="sum")
    twr = team.index.get_level_values("team_id").map(team_wr).to_numpy(dtype=float)
    team["team_games"] = np.log1p(team["games"])
    team["team_edge"] = (team["wins"] + cfg.team_prior * twr) / (team["games"] + cfg.team_prior) - twr

    rp = recent[recent["player_id"].notna()]
    pg = rp.groupby("player_id")["won"].agg(["size", "sum"])
    player_wr = (pg["sum"] + cfg.player_prior * 0.5) / (pg["size"] + cfg.player_prior)
    player = rp.groupby(["player_id", "champion_id"])["won"].agg(games="size", wins="sum")
    pwr = player.index.get_level_values("player_id").map(player_wr).to_numpy(dtype=float)
    player["player_games"] = np.log1p(player["games"])
    player["player_edge"] = (player["wins"] + cfg.player_prior * pwr) / (player["games"] + cfg.player_prior) - pwr

    m = recent.loc[recent["role"].isin(ROLES), ["game_id", "team_id", "role", "champion_id", "won"]]
    mm = m.merge(m, on=["game_id", "role"], suffixes=("", "_opp"))
    mm = mm[mm["team_id"] != mm["team_id_opp"]]
    matchup = (mm.groupby(["role", "champion_id", "champion_id_opp"])["won"].agg(games="size", wins="sum")
               .rename_axis(["role", "champion_id", "opp_champion_id"]))
    smoothed = (matchup["wins"] + cfg.matchup_prior * 0.5) / (matchup["games"] + cfg.matchup_prior) - 0.5
    matchup["edge"] = smoothed.where(matchup["games"] >= cfg.matchup_min_games, 0.0)

    return History(champ=champ, champ_roles=recent.groupby(["champion_id", "role"]).size(),
                   team=team, team_all=recent.groupby(["team_id", "champion_id"]).size(),
                   player=player, matchup=matchup, window=list(window))


def contributions(rows: pd.DataFrame, hist: History) -> pd.DataFrame:
    """Per-action feature values for rows with team_id, opponent_id, action, champion_id,
    role and player_id. Picks fill PICK_COLS, bans fill `denied`; unseen -> 0."""
    out = rows.copy()
    pick = (out["action"] == "pick").to_numpy()
    c = hist.champ.reindex(out["champion_id"])
    t = hist.team.reindex(pd.MultiIndex.from_arrays([out["team_id"], out["champion_id"], out["role"]]))
    p = hist.player.reindex(pd.MultiIndex.from_arrays([out["player_id"].fillna(""), out["champion_id"]]))
    o = hist.team_all.reindex(pd.MultiIndex.from_arrays([out["opponent_id"], out["champion_id"]]))
    for col, src in (("champ_wr", c["champ_wr"]), ("presence", c["presence"]), ("priority", c["priority"]),
                     ("team_games", t["team_games"]), ("team_edge", t["team_edge"]),
                     ("player_games", p["player_games"]), ("player_edge", p["player_edge"])):
        out[col] = np.where(pick, src.fillna(0).to_numpy(), 0.0)
    out["denied"] = np.where(~pick, np.log1p(o.fillna(0).to_numpy()), 0.0)
    out["n_picks"], out["n_bans"] = pick.astype(float), (~pick).astype(float)
    # sample sizes behind the numbers, for display
    out["champ_picks"] = c["picks"].fillna(0).to_numpy()
    out["champ_wins"] = c["wins"].fillna(0).to_numpy()
    out["window_games"] = int(hist.champ["window_games"].iloc[0]) if len(hist.champ) else 0
    out["team_games_n"] = t["games"].fillna(0).to_numpy()
    out["team_wins_n"] = t["wins"].fillna(0).to_numpy()
    out["player_games_n"] = p["games"].fillna(0).to_numpy()
    out["player_wins_n"] = p["wins"].fillna(0).to_numpy()
    return out


def matchup_rows(picks: pd.DataFrame, hist: History) -> pd.DataFrame:
    """One row per (game, role) where both sides have a pick in that role: blue's
    smoothed edge vs red's champion, the prior games behind it, and the draft stage at
    which both picks exist."""
    p = picks[picks["role"].isin(ROLES)]
    m = p[p["side"] == "blue"].merge(p[p["side"] == "red"], on=["game_id", "role"], suffixes=("_b", "_r"))
    e = hist.matchup.reindex(pd.MultiIndex.from_arrays([m["role"], m["champion_id_b"], m["champion_id_r"]]))
    m["matchup"] = e["edge"].fillna(0).to_numpy()
    m["matchup_games"] = e["games"].fillna(0).to_numpy()
    m["matchup_wins"] = e["wins"].fillna(0).to_numpy()
    m["stage"] = np.maximum(m["draft_order_b"], m["draft_order_r"])
    return m[["game_id", "role", "champion_id_b", "champion_id_r", "matchup", "matchup_games", "matchup_wins", "stage"]]


# --- feature table ------------------------------------------------------------------------
def build_features(drafts: pd.DataFrame, cfg: FeatureConfig = FeatureConfig(),
                   stages: list[int] = STAGES) -> pd.DataFrame:
    """One row per (game, stage); stage = draft actions completed. Blue's point of view."""
    games, actions = prepare(drafts)
    pre_b, pre_r, _ = elo(games, cfg.elo_k, cfg.elo_init)

    contrib, mu = [], []
    for day, day_actions in actions.groupby("date", sort=True):
        hist = history_before(games, actions, day, cfg)
        contrib.append(contributions(day_actions, hist))
        mu.append(matchup_rows(day_actions[day_actions["action"] == "pick"], hist))
    contrib, mu = pd.concat(contrib, ignore_index=True), pd.concat(mu, ignore_index=True)

    blue_idx = pd.MultiIndex.from_arrays([games.index, games["blue_team"]])
    red_idx = pd.MultiIndex.from_arrays([games.index, games["red_team"]])
    base = games[["date", "patch", "league", "series_id", "blue_team", "red_team", "blue_first_pick", "blue_won"]].copy()
    base["elo_diff"] = pre_b - pre_r
    parts = []
    for s in stages:
        sums = contrib[contrib["draft_order"] <= s].groupby(["game_id", "team_id"])[DRAFT_COLS].sum()
        part = base.copy()
        part[[f"d_{c}" for c in DRAFT_COLS]] = (sums.reindex(blue_idx).fillna(0).to_numpy()
                                                - sums.reindex(red_idx).fillna(0).to_numpy())
        part["matchup"] = mu[mu["stage"] <= s].groupby("game_id")["matchup"].sum().reindex(games.index).fillna(0).to_numpy()
        part["stage"] = s
        parts.append(part)
    out = pd.concat(parts).rename_axis("game_id").reset_index()
    return out[["game_id", "stage", "date", "patch", "league", "series_id", "blue_team", "red_team", "blue_won"]
               + FEATURES]


# --- models, calibration, metrics ---------------------------------------------------------
def make_model(kind: str, C: float = DEFAULT_C):
    """kind: "elo" (Elo only, no intercept, so no side information), "side" (Elo + side:
    the intercept is blue's side advantage, plus blue_first_pick), "full" (everything)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "elo":
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(fit_intercept=False))
    if kind == "side":
        return make_pipeline(StandardScaler(), LogisticRegression())
    if kind == "full":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    raise ValueError(kind)


KIND_FEATURES = {"elo": ELO_FEATURES, "side": SIDE_FEATURES, "full": FEATURES}


def _weights(feats: pd.DataFrame) -> np.ndarray:
    """Each game counts once in total, however many of its stages are present."""
    return (1.0 / feats.groupby("game_id")["stage"].transform("size")).to_numpy()


def fit(kind: str, feats: pd.DataFrame, C: float = DEFAULT_C):
    if kind != "full":                       # these features don't change during the draft
        feats = feats[feats["stage"] == FULL]
    model = make_model(kind, C)
    model.fit(feats[KIND_FEATURES[kind]].to_numpy(), feats["blue_won"].to_numpy(),
              logisticregression__sample_weight=_weights(feats))
    return model


def predict(model, feats: pd.DataFrame, kind: str) -> np.ndarray:
    return model.predict_proba(feats[KIND_FEATURES[kind]].to_numpy())[:, 1]


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(p, y, w=None, intercept: bool = True) -> tuple[float, float]:
    """Platt scaling: P = sigmoid(a * logit(p) + b). Without intercept, b = 0 (keeps
    the Elo-only model free of side information)."""
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1e6, fit_intercept=intercept)
    lr.fit(_logit(p).reshape(-1, 1), np.asarray(y), sample_weight=w)
    return float(lr.coef_[0, 0]), (float(lr.intercept_[0]) if intercept else 0.0)


def platt_apply(cal: tuple[float, float], p) -> np.ndarray:
    a, b = cal
    return 1 / (1 + np.exp(-(a * _logit(p) + b)))


def log_loss(y, p) -> float:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _per_game(y, p) -> dict[str, np.ndarray]:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"accuracy": ((p >= 0.5) == y).astype(float),
            "log_loss": -(y * np.log(p) + (1 - y) * np.log(1 - p)),
            "brier": (p - y) ** 2}


def _bootstrap_weights(series: pd.Series, n: int = N_BOOT, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Resample whole series (games in one series aren't independent). Returns (codes, W)
    where W[i, s] = how often series s is drawn in bootstrap sample i."""
    codes, uniq = pd.factorize(series)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(uniq), size=(n, len(uniq)))
    W = np.zeros((n, len(uniq)))
    np.add.at(W, (np.repeat(np.arange(n), len(uniq)), draws.ravel()), 1)
    return codes, W


def _boot_mean(values: np.ndarray, codes: np.ndarray, W: np.ndarray) -> np.ndarray:
    S = W.shape[1]
    return (W @ np.bincount(codes, weights=values, minlength=S)) / (W @ np.bincount(codes, minlength=S))


def scores_with_ci(df: pd.DataFrame, cols: dict[str, str], boot=None) -> pd.DataFrame:
    """Accuracy / log loss / Brier with 95% series-bootstrap CIs for each prediction column."""
    codes, W = boot or _bootstrap_weights(df["series_id"])
    rows = []
    for name, col in cols.items():
        row = {"model": name, "games": len(df), "series": int(df["series_id"].nunique())}
        for metric, v in _per_game(df["blue_won"], df[col]).items():
            lo, hi = np.percentile(_boot_mean(v, codes, W), [2.5, 97.5])
            row |= {metric: float(v.mean()), f"{metric}_lo": float(lo), f"{metric}_hi": float(hi)}
        rows.append(row)
    return pd.DataFrame(rows)


def gain_with_ci(df: pd.DataFrame, base: str, new: str, boot=None) -> dict:
    """Improvement of `new` over `base` (positive = better), with 95% series-bootstrap CIs
    and the share of bootstrap samples with no improvement."""
    codes, W = boot or _bootstrap_weights(df["series_id"])
    b, n = _per_game(df["blue_won"], df[base]), _per_game(df["blue_won"], df[new])
    out = {}
    for metric, sign in (("accuracy", 1), ("log_loss", -1), ("brier", -1)):
        diff = sign * (n[metric] - b[metric])
        boots = _boot_mean(diff, codes, W)
        out[metric] = {"gain": float(diff.mean()), "ci": [float(x) for x in np.percentile(boots, [2.5, 97.5])],
                       "p_no_gain": float(np.mean(boots <= 0))}
    return out


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    return (df.groupby("bin").agg(games=("y", "size"), p_low=("p", "min"), p_high=("p", "max"),
                                  predicted=("p", "mean"), actual=("y", "mean")).reset_index(drop=True))


# --- walk-forward validation --------------------------------------------------------------
def walk_forward(feats: pd.DataFrame, start: str = WARMUP_START, c_grid=C_GRID) -> list[dict]:
    """Monthly folds from `start`. For each month: fit on every earlier game, predict
    the month. C for the full model and the Platt calibrators are then chosen per month
    from EARLIER months' out-of-sample predictions only (the first month uses DEFAULT_C,
    and months with fewer than MIN_CALIBRATION_GAMES earlier test games stay uncalibrated)."""
    folds = []
    for m0 in pd.date_range(pd.Timestamp(start), feats["date"].max(), freq="MS"):
        m1 = m0 + pd.offsets.MonthBegin(1)
        test = feats[(feats["date"] >= m0) & (feats["date"] < m1)]
        train = feats[feats["date"] < m0]
        if test.empty or train["game_id"].nunique() < 50:
            continue
        preds = test[["game_id", "stage", "date", "league", "series_id", "blue_won"]].copy()
        models = {"elo": fit("elo", train), "side": fit("side", train)}
        preds["elo"], preds["side"] = predict(models["elo"], test, "elo"), predict(models["side"], test, "side")
        for C in c_grid:
            models[("full", C)] = fit("full", train, C)
            preds[f"full_{C}"] = predict(models[("full", C)], test, "full")
        folds.append({"month": m0, "preds": preds, "models": models, "train_games": int(train["game_id"].nunique())})

    for i, f in enumerate(folds):
        prev = pd.concat([folds[j]["preds"] for j in range(i)]) if i else None
        if prev is None:
            C = DEFAULT_C
        else:
            pf = prev[prev["stage"] == FULL]
            C = min(c_grid, key=lambda c: log_loss(pf["blue_won"], pf[f"full_{c}"]))
        p = f["preds"]
        f["C"] = C
        p["full_raw"] = p[f"full_{C}"]
        enough = prev is not None and prev["game_id"].nunique() >= MIN_CALIBRATION_GAMES
        f["calibrated"] = bool(enough)
        f["calibrators"] = {}
        for name, src, intercept in (("elo", "elo", False), ("side", "side", True), ("full", f"full_{C}", True)):
            own = "full_raw" if name == "full" else src
            if enough:
                cal = platt_fit(prev[src], prev["blue_won"], _weights(prev), intercept)
                f["calibrators"][name] = cal
                p[f"{name}_cal"] = platt_apply(cal, p[own])
            else:
                p[f"{name}_cal"] = p[own]
    return folds


def permutation_importance(folds: list[dict], feats: pd.DataFrame, repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    """Rise in full-draft log loss on each month's test games when one feature (or the
    whole draft block) is shuffled, using that month's own model. Pooled over months."""
    rng = np.random.default_rng(seed)
    groups = {f: [f] for f in FEATURES if f not in ("d_n_picks", "d_n_bans")}
    groups["ALL draft features"] = DRAFT_FEATURES
    total = {g: 0.0 for g in groups}
    n = 0
    for f in folds:
        test = feats[feats["game_id"].isin(set(f["preds"]["game_id"])) & (feats["stage"] == FULL)]
        model, y = f["models"][("full", f["C"])], test["blue_won"].to_numpy()
        base = log_loss(y, predict(model, test, "full"))
        for g, cols in groups.items():
            rises = []
            for _ in range(repeats):
                shuffled = test.copy()
                shuffled[cols] = test[cols].to_numpy()[rng.permutation(len(test))]
                rises.append(log_loss(y, predict(model, shuffled, "full")) - base)
            total[g] += np.mean(rises) * len(test)
        n += len(test)
    return (pd.Series({g: v / n for g, v in total.items()}, name="log_loss_rise")
            .sort_values(ascending=False).rename_axis("feature").reset_index())


def evaluate(drafts: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """Walk-forward report plus the final model, refitted on all games, for live use."""
    log = print if verbose else (lambda *a, **k: None)
    df = load_drafts() if drafts is None else drafts
    games, _ = prepare(df)
    k, k_table = tune_elo_k(games)
    cfg = FeatureConfig(elo_k=k)
    log(f"Elo K = {k} (tuned on {int(k_table['games'].iloc[0])} games before {REPORT_START})")
    feats = build_features(df, cfg)
    folds = walk_forward(feats)
    reported = [f for f in folds if f["month"] >= pd.Timestamp(REPORT_START)]
    pooled = pd.concat([f["preds"] for f in reported], ignore_index=True)
    full = pooled[pooled["stage"] == FULL]
    log(f"{len(reported)} reported monthly folds, {full['game_id'].nunique()} test games")

    boot = _bootstrap_weights(full["series_id"])
    cols = {"Elo only": "elo_cal", "Elo + side": "side_cal", "Full model": "full_cal",
            "Full model, uncalibrated": "full_raw"}
    overall = scores_with_ci(full, cols, boot)
    gains = {"full_vs_side": gain_with_ci(full, "side_cal", "full_cal", boot),
             "full_vs_elo": gain_with_ci(full, "elo_cal", "full_cal", boot),
             "side_vs_elo": gain_with_ci(full, "elo_cal", "side_cal", boot)}
    league_cols = {k: v for k, v in cols.items() if "uncal" not in k}
    leagues = pd.concat([scores_with_ci(part, league_cols).assign(league=lg)
                         for lg, part in full.groupby("league")], ignore_index=True)
    league_gains = {lg: gain_with_ci(part, "side_cal", "full_cal") for lg, part in full.groupby("league")}
    stage_rows = []
    for s in (0, 6, 9, 12, 16, 18, 20):
        part = pooled[pooled["stage"] == s]
        stage_rows.append({"stage": s, **{m: float(v.mean()) for m, v in _per_game(part["blue_won"], part["full_cal"]).items()}})
    p0 = pooled[pooled["stage"] == 0].set_index("game_id")["full_cal"]
    p20 = full.set_index("game_id")["full_cal"]
    shift = (p20 - p0.reindex(p20.index)).abs()

    importance = permutation_importance(reported, feats)
    all_oos = pd.concat([f["preds"] for f in folds])
    all_full = all_oos[all_oos["stage"] == FULL]
    C_final = min(C_GRID, key=lambda c: log_loss(all_full["blue_won"], all_full[f"full_{c}"]))
    final = fit("full", feats, C_final)
    calibrator = platt_fit(all_oos[f"full_{C_final}"], all_oos["blue_won"], _weights(all_oos))
    lr = final.steps[-1][1]
    coefs = pd.DataFrame({"feature": FEATURES, "coef_std": lr.coef_[0]}).sort_values(
        "coef_std", key=np.abs, ascending=False)
    full_feats = feats[feats["game_id"].isin(full["game_id"]) & (feats["stage"] == FULL)]

    report = {
        "config": asdict(cfg), "elo_k_table": k_table.to_dict("records"),
        "folds": [{"month": str(f["month"].date()), "train_games": f["train_games"],
                   "test_games": int(f["preds"]["game_id"].nunique()), "C": f["C"], "calibrated": f["calibrated"],
                   "reported": bool(f["month"] >= pd.Timestamp(REPORT_START))} for f in folds],
        "overall": overall.to_dict("records"), "gains": gains,
        "leagues": leagues.to_dict("records"), "league_gains": league_gains,
        "by_stage": stage_rows,
        "draft_shift": {"median": float(shift.median()), "p90": float(shift.quantile(.9)), "max": float(shift.max())},
        "calibration": {"Elo + side": calibration_table(full["blue_won"], full["side_cal"]).to_dict("records"),
                        "Full model": calibration_table(full["blue_won"], full["full_cal"]).to_dict("records"),
                        "Full model, uncalibrated": calibration_table(full["blue_won"], full["full_raw"]).to_dict("records")},
        "importance": importance.to_dict("records"),
        "coefficients": coefs.to_dict("records"),
        "intercept": float(lr.intercept_[0]),
        "final": {"C": C_final, "platt": list(calibrator), "games": int(len(games)),
                  "through": str(games["date"].max().date())},
        "sides": {"test_games": int(len(full)), "blue_win_rate": float(full["blue_won"].mean()),
                  "blue_first_pick_share": float(full_feats["blue_first_pick"].mean()),
                  "blue_win_rate_blue_fp": float(full_feats.loc[full_feats["blue_first_pick"] == 1, "blue_won"].mean()),
                  "blue_win_rate_red_fp": float(full_feats.loc[full_feats["blue_first_pick"] == 0, "blue_won"].mean()),
                  "games_red_fp": int((full_feats["blue_first_pick"] == 0).sum())},
    }
    return {"model": final, "calibrator": calibrator, "features": FEATURES, "config": asdict(cfg),
            "trained_through": report["final"]["through"], "report": report}


def save_model(bundle: dict, path: Path = MODEL_PATH) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(rebuild: bool = False, path: Path = MODEL_PATH) -> dict:
    """The saved model; rebuilt if missing, stale (different features) or rebuild=True."""
    import joblib
    if not rebuild and path.exists():
        bundle = joblib.load(path)
        if bundle.get("features") == FEATURES and "calibrator" in bundle:
            return bundle
    bundle = evaluate(verbose=False)
    save_model(bundle, path)
    return bundle


# --- live / partial-draft prediction ------------------------------------------------------
def _resolve_team(x: str, drafts: pd.DataFrame) -> str:
    if str(x) in set(drafts["team_id"].astype(str)):
        return str(x)
    names = drafts.sort_values("date").groupby("team_id")["team_name"].last().astype(str)
    hit = names[names.str.lower() == str(x).lower()]
    if hit.empty:
        raise ValueError(f"unknown team: {x!r}")
    return str(hit.index[0])


def _resolve_champions(xs, lookup: dict[str, str], what: str) -> list[str]:
    out = []
    for x in xs or ():
        if str(x).lower() not in lookup:
            raise ValueError(f"unknown champion in {what}: {x!r}")
        out.append(lookup[str(x).lower()])
    return out


def guess_roles(team_id: str, champs: list[str], hist: History) -> list[str]:
    """Most likely distinct roles for a team's picks, from how often this team and all
    teams played each champion in each role before the date."""
    if not champs:
        return []
    team_counts = hist.team["games"]
    score = np.array([[np.log(team_counts.get((team_id, c, r), 0) + 0.5)
                       + np.log(hist.champ_roles.get((c, r), 0) + 0.5) for r in ROLES] for c in champs])
    best = max(permutations(range(len(ROLES)), len(champs)),
               key=lambda perm: score[np.arange(len(champs)), list(perm)].sum())
    return [ROLES[i] for i in best]


def latest_lineup(actions: pd.DataFrame, team_id: str, before) -> dict[str, str]:
    """role -> player_id from the team's most recent game before `before`."""
    a = actions[(actions["team_id"] == team_id) & (actions["action"] == "pick")
                & (actions["date"] < pd.Timestamp(before)) & actions["player_id"].notna()]
    if a.empty:
        return {}
    last = a[a["game_id"] == a.sort_values("date", kind="stable")["game_id"].iloc[-1]]
    return dict(zip(last["role"], last["player_id"]))


def predict_draft(team_a: str, team_b: str, side_a: str, picks_a=(), picks_b=(), bans=(),
                  date=None, patch: str | None = None, first_pick: str | None = None,
                  roles_a=None, roles_b=None, model: dict | None = None,
                  drafts: pd.DataFrame | None = None) -> dict:
    """Calibrated P(team_a wins) for a draft in progress (0-5 picks per side).

    team_a/team_b: team_id or name. side_a: "blue" or "red".
    picks_a/picks_b: champion names or ids so far, in pick order.
    bans: flat list (who banned what unknown: only checked for conflicts) or
          {"a": [...], "b": [...]}, which also feeds `d_denied`.
    roles_a/roles_b: optional role per pick; otherwise guessed from history. The player
          for a role is taken from the team's most recent lineup.
    date: only games strictly before it are used (default: day after the last game).
    patch: optional end of the meta window. first_pick: "a" or "b" (default: blue).
    """
    df = load_drafts() if drafts is None else drafts
    bundle = load_model() if model is None else model
    cfg = FeatureConfig(**bundle["config"])
    if side_a not in ("blue", "red"):
        raise ValueError("side_a must be 'blue' or 'red'")
    a_id, b_id = _resolve_team(team_a, df), _resolve_team(team_b, df)
    if a_id == b_id:
        raise ValueError("team_a and team_b are the same team")

    names = df.drop_duplicates("champion_id").set_index("champion_id")["champion_name"].astype(str)
    lookup = {str(i).lower(): str(i) for i in names.index} | {n.lower(): str(i) for i, n in names.items()}
    pa, pb = _resolve_champions(picks_a, lookup, "picks_a"), _resolve_champions(picks_b, lookup, "picks_b")
    if isinstance(bans, dict):
        if set(bans) - {"a", "b"}:
            raise ValueError(f"bans dict keys must be 'a'/'b', got {sorted(set(bans) - {'a', 'b'})}")
        ba, bb = _resolve_champions(bans.get("a"), lookup, "bans"), _resolve_champions(bans.get("b"), lookup, "bans")
        flat = ba + bb
    else:
        ba, bb, flat = [], [], _resolve_champions(bans, lookup, "bans")
    if len(pa) > 5 or len(pb) > 5:
        raise ValueError("at most 5 picks per side")
    if len(flat) > 10:
        raise ValueError("at most 10 bans")
    used = pa + pb + flat
    if len(set(used)) != len(used):
        raise ValueError(f"champion used more than once in this draft: {sorted({names[c] for c in used if used.count(c) > 1})}")
    for roles, picks, what in ((roles_a, pa, "roles_a"), (roles_b, pb, "roles_b")):
        if roles is not None and (len(roles) != len(picks) or len(set(roles)) != len(roles)
                                  or not set(roles) <= set(ROLES)):
            raise ValueError(f"{what} must give one distinct role from {ROLES} per pick")

    games, actions = prepare(df)
    cutoff = games["date"].max() + pd.Timedelta(days=1) if date is None else pd.Timestamp(date)
    prior = games[games["date"] < cutoff]
    _, _, ratings = elo(prior, cfg.elo_k, cfg.elo_init)
    hist = history_before(games, actions, cutoff, cfg, patch)

    side_b = "red" if side_a == "blue" else "blue"
    ra = list(roles_a) if roles_a is not None else guess_roles(a_id, pa, hist)
    rb = list(roles_b) if roles_b is not None else guess_roles(b_id, pb, hist)
    lineup = {a_id: latest_lineup(actions, a_id, cutoff), b_id: latest_lineup(actions, b_id, cutoff)}
    a_first = (side_a == "blue") if first_pick is None else (first_pick == "a")
    rows = ([(a_id, b_id, side_a, "pick", c, r, lineup[a_id].get(r)) for c, r in zip(pa, ra)]
            + [(b_id, a_id, side_b, "pick", c, r, lineup[b_id].get(r)) for c, r in zip(pb, rb)]
            + [(a_id, b_id, side_a, "ban", c, "?", None) for c in ba]
            + [(b_id, a_id, side_b, "ban", c, "?", None) for c in bb])
    rows = pd.DataFrame(rows, columns=["team_id", "opponent_id", "side", "action", "champion_id", "role", "player_id"])
    rows = rows.assign(game_id="live", draft_order=0)
    contrib = contributions(rows, hist)
    mu = matchup_rows(rows[rows["action"] == "pick"], hist)

    blue_id, red_id = (a_id, b_id) if side_a == "blue" else (b_id, a_id)
    sums = contrib.groupby("team_id")[DRAFT_COLS].sum().reindex([blue_id, red_id]).fillna(0)
    x = {"elo_diff": ratings.get(blue_id, cfg.elo_init) - ratings.get(red_id, cfg.elo_init),
         "blue_first_pick": int(a_first == (side_a == "blue"))}
    x |= {f"d_{c}": float(sums.loc[blue_id, c] - sums.loc[red_id, c]) for c in DRAFT_COLS}
    x["matchup"] = float(mu["matchup"].sum())
    X = pd.DataFrame([x])[bundle["features"]].to_numpy()
    raw = float(bundle["model"].predict_proba(X)[0, 1])
    p_blue = float(platt_apply(bundle["calibrator"], raw))
    p_a = p_blue if side_a == "blue" else 1 - p_blue

    scaler, lr = bundle["model"].steps[0][1], bundle["model"].steps[-1][1]
    sign = 1 if side_a == "blue" else -1
    # Each feature's push on team_a's log-odds vs an average training game, after calibration.
    push = (X[0] - scaler.mean_) / scaler.scale_ * lr.coef_[0] * bundle["calibrator"][0] * sign

    contrib = contrib.assign(champion=contrib["champion_id"].map(names), matchup_vs=None,
                             matchup_games=0.0, matchup_wins=0.0)
    for _, m in mu.iterrows():   # attach each matchup to both picks, from that pick's side
        for champ, opp, wins in ((m["champion_id_b"], m["champion_id_r"], m["matchup_wins"]),
                                 (m["champion_id_r"], m["champion_id_b"], m["matchup_games"] - m["matchup_wins"])):
            i = contrib.index[(contrib["champion_id"] == champ) & (contrib["action"] == "pick")]
            contrib.loc[i, ["matchup_vs", "matchup_games", "matchup_wins"]] = [names.get(opp), m["matchup_games"], wins]
    return {
        "team_a": a_id, "team_b": b_id, "blue": blue_id, "first_pick": a_id if a_first else b_id,
        "p_team_a": p_a, "p_team_b": 1 - p_a, "p_blue": p_blue,
        "p_team_a_elo_only": float(elo_prob(sign * x["elo_diff"])),
        "elo": {"team_a": ratings.get(a_id, cfg.elo_init), "team_b": ratings.get(b_id, cfg.elo_init)},
        "features_blue_view": x,
        "push": dict(zip(bundle["features"], map(float, push))),
        "roles": {"a": ra, "b": rb},
        "history_before": str(cutoff.date()), "history_games": int(len(prior)),
        "champion_window": hist.window,
        "actions": contrib,
    }


# --- CLI ----------------------------------------------------------------------------------
def _ci(row, metric, pct=False) -> str:
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.4f}")
    return f"{f(row[metric])} ({f(row[metric + '_lo'])}-{f(row[metric + '_hi'])})"


def format_report(r: dict) -> str:
    out = []
    rep = [f for f in r["folds"] if f["reported"]]
    out.append(f"Elo K {r['config']['elo_k']}; {len(rep)} reported monthly folds {rep[0]['month']}..{rep[-1]['month']}")
    out.append("Folds: " + ", ".join(f"{f['month'][:7]} n={f['test_games']} C={f['C']}{'' if f['calibrated'] else ' (uncal)'}"
                                     for f in r["folds"]))
    out.append("\n| Model | Games | Accuracy | Log loss | Brier |\n|---|---|---|---|---|")
    for row in r["overall"]:
        out.append(f"| {row['model']} | {row['games']} | {_ci(row, 'accuracy', True)} | {_ci(row, 'log_loss')} | {_ci(row, 'brier')} |")
    for name, g in list(r["gains"].items()) + [(f"league {k}: full_vs_side", v) for k, v in r["league_gains"].items()]:
        out.append(f"{name}: " + "; ".join(
            f"{m} {v['gain']:+.4f} (95% CI {v['ci'][0]:+.4f}..{v['ci'][1]:+.4f}, P(no gain)={v['p_no_gain']:.3f})"
            for m, v in g.items()))
    out.append("\n| League | Model | Games | Accuracy | Log loss | Brier |\n|---|---|---|---|---|---|")
    for row in r["leagues"]:
        out.append(f"| {row['league']} | {row['model']} | {row['games']} | {_ci(row, 'accuracy', True)} | "
                   f"{_ci(row, 'log_loss')} | {_ci(row, 'brier')} |")
    out.append("\n| Stage | Accuracy | Log loss | Brier |\n|---|---|---|---|")
    for row in r["by_stage"]:
        out.append(f"| {row['stage']} | {row['accuracy']:.1%} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    out.append(f"Draft shift stage 0 -> 20: {r['draft_shift']}")
    for name, rows in r["calibration"].items():
        out.append(f"\nCalibration, {name}:\n| Predicted | Games | Mean predicted | Actual |\n|---|---|---|---|")
        for row in rows:
            out.append(f"| {row['p_low']:.0%}-{row['p_high']:.0%} | {row['games']} | {row['predicted']:.1%} | {row['actual']:.1%} |")
    out.append("\nPermutation importance (log-loss rise):")
    out += [f"  {row['feature']:22s} {row['log_loss_rise']:+.5f}" for row in r["importance"]]
    out.append(f"\nFinal model (C={r['final']['C']}, Platt {r['final']['platt']}), intercept {r['intercept']:+.3f}, coefficients:")
    out += [f"  {row['feature']:22s} {row['coef_std']:+.4f}" for row in r["coefficients"]]
    out.append(f"\nSides: {r['sides']}")
    out.append("Elo K table:\n" + pd.DataFrame(r["elo_k_table"]).to_string(index=False))
    return "\n".join(out)


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    bundle = evaluate()
    print(format_report(bundle["report"]))
    save_model(bundle)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
