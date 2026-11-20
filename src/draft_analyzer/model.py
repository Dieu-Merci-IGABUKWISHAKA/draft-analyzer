"""Draft win-probability model: P(first-pick team wins) from team strength and the draft.

One row per game, from the point of view of the team that picks first
(slot 7). Every feature for a game is computed only from games played on
EARLIER DATES (strictly before the game's date), so nothing from the game
itself, later games the same day, or the future leaks in.

Features (all "first-pick team minus second-pick team" unless noted):
  elo_diff       Elo rating difference, ratings updated game by game (K tuned on early data)
  fp_blue        1 if the first-pick team is on blue side (since 2026 it can be red)
  d_champ_wr     sum over picks of the champion's smoothed win rate - 50%, over all teams
                 in the last `champ_patches` patches seen before the game
  d_presence     sum over picks of champion presence (picked or banned) in that window
  d_comfort_games  sum over picks of log(1 + the team's games on that champion in its usual role
                 for it), over the team's last `comfort_days` days
  d_comfort_edge sum over picks of the team's smoothed win rate on that champion minus
                 its own overall win rate
  d_denied       sum over the team's bans of log(1 + games the OPPONENT played on that champion)
  d_n_picks, d_n_bans  how many picks / bans each side has made so far

Sums (not averages) let the same model score partial drafts: a pick not made
yet adds nothing. The model is trained on every stage of each draft (after
0, 1, ..., 20 draft actions), each stage weighted 1/21, so it is calibrated
mid-draft too. See docs/MODEL.md for validation results.

Usage:
    python src/draft_analyzer/model.py            # validate, print report, save the model
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from datetime import date as Date
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer.analysis import FIRST_PICK_SLOT, _patch_key  # noqa: E402
from draft_analyzer.config import CACHE_DIR  # noqa: E402
from draft_analyzer.drafts import load_drafts  # noqa: E402

MODEL_PATH = CACHE_DIR / "draft_model.joblib"

TEST_START = "2026-04-01"        # test = games on/after this date; never used to fit or tune
INNER_VAL_START = "2026-01-01"   # model choice: fit before this, compare on [this, TEST_START)
ELO_BURN_IN_DAYS = 60            # first days of data skipped when scoring K (all ratings still 1500)
ELO_K_GRID = [8, 12, 16, 20, 24, 32, 40, 48, 64, 80]
LR_C_GRID = [0.01, 0.1, 1.0, 10.0]

STAGES = list(range(0, 21))      # draft actions completed: 0 = nothing yet, 20 = full draft
FULL = 20
DRAFT_COLS = ["champ_wr", "presence", "comfort_games", "comfort_edge", "denied", "n_picks", "n_bans"]
BASELINE_FEATURES = ["elo_diff"]
SIDE_FEATURES = ["elo_diff", "fp_blue"]
FEATURES = ["elo_diff", "fp_blue"] + [f"d_{c}" for c in DRAFT_COLS]

FEATURE_LABELS = {
    "elo_diff": "Team strength (Elo)",
    "fp_blue": "Side (first-pick team on blue)",
    "d_champ_wr": "Champion win rates this patch window",
    "d_presence": "Champion priority (pick/ban presence)",
    "d_comfort_games": "Team experience on its picks",
    "d_comfort_edge": "Team's own results on its picks",
    "d_denied": "Bans that hit the opponent's pool",
    "d_n_picks": "Picks made so far",
    "d_n_bans": "Bans made so far",
}


@dataclass(frozen=True)
class FeatureConfig:
    """Fixed before looking at results (not tuned), except elo_k which is tuned on early data."""
    elo_k: float = 24.0
    elo_init: float = 1500.0
    champ_patches: int = 3        # champion stats: last N patches with games before the game
    champ_prior: float = 20.0     # smoothing: imaginary games at 50% per champion
    comfort_days: int = 365       # team comfort: the team's games in the last N days
    comfort_prior: float = 5.0    # smoothing: imaginary games at the team's own win rate


# --- tables -------------------------------------------------------------------------------
def prepare(drafts: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, actions). games: one row per game, indexed by game_id, sorted by
    date/series/game_number, with the first-pick (fp) and second-pick (sp) teams.
    actions: one row per pick/ban with plain str columns and `won` for the drafting team."""
    d = drafts.copy()
    for col in ("patch", "side", "action", "role_inferred", "event"):
        d[col] = d[col].astype(object).where(d[col].notna(), None)
        d[col] = d[col].map(lambda v: None if v is None else str(v))
    d["date"] = pd.to_datetime(d["date"])

    g = d.drop_duplicates("game_id").set_index("game_id")[
        ["series_id", "game_number", "event", "date", "patch", "game_winner_team_id"]]
    fp = d[(d["action"] == "pick") & (d["draft_order"] == FIRST_PICK_SLOT)].set_index("game_id")
    g = g.join(fp[["team_id", "opponent_id", "side"]].rename(
        columns={"team_id": "fp_team", "opponent_id": "sp_team", "side": "fp_side"}), how="inner")
    g = g[g["game_winner_team_id"].notna()]
    g["fp_won"] = (g["game_winner_team_id"] == g["fp_team"]).astype(int)
    g["fp_blue"] = (g["fp_side"] == "blue").astype(int)
    g = g.sort_values(["date", "series_id", "game_number"])

    a = d[d["game_id"].isin(g.index)][["game_id", "date", "patch", "team_id", "opponent_id", "action",
                                       "draft_order", "champion_id", "role_inferred"]].copy()
    a = a.rename(columns={"role_inferred": "role"})
    a["role"] = a["role"].fillna("?")
    a["won"] = (a["team_id"] == a["game_id"].map(g["game_winner_team_id"])).astype(int)
    return g, a.reset_index(drop=True)


# --- Elo ----------------------------------------------------------------------------------
def elo(games: pd.DataFrame, k: float, init: float = 1500.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Ratings of (fp, sp) before each game, frozen for a whole date: games on the same
    date all see the ratings from the end of the previous date. Also returns the final
    ratings. `games` must be sorted by date (prepare() does this)."""
    ratings: dict[str, float] = {}
    dates, fp, sp, won = (games["date"].to_numpy(), games["fp_team"].to_numpy(),
                          games["sp_team"].to_numpy(), games["fp_won"].to_numpy())
    pre_fp, pre_sp = np.empty(len(games)), np.empty(len(games))
    i, n = 0, len(games)
    while i < n:
        j = i
        while j < n and dates[j] == dates[i]:
            pre_fp[j], pre_sp[j] = ratings.get(fp[j], init), ratings.get(sp[j], init)
            j += 1
        for t in range(i, j):
            ra, rb = ratings.get(fp[t], init), ratings.get(sp[t], init)
            delta = k * (won[t] - 1 / (1 + 10 ** ((rb - ra) / 400)))
            ratings[fp[t]], ratings[sp[t]] = ra + delta, rb - delta
        i = j
    return pre_fp, pre_sp, ratings


def elo_prob(diff) -> np.ndarray:
    return 1 / (1 + 10 ** (-np.asarray(diff, dtype=float) / 400))


def tune_elo_k(games: pd.DataFrame, before: str = INNER_VAL_START, grid=ELO_K_GRID,
               init: float = 1500.0) -> tuple[float, pd.DataFrame]:
    """Pick K by one-step-ahead log loss of raw Elo on games before `before` (after a burn-in)."""
    g = games[games["date"] < pd.Timestamp(before)]
    scored = g["date"] >= g["date"].min() + pd.Timedelta(days=ELO_BURN_IN_DAYS)
    rows = []
    for k in grid:
        pre_fp, pre_sp, _ = elo(g, k, init)
        p = np.clip(elo_prob(pre_fp - pre_sp), 1e-6, 1 - 1e-6)[scored.to_numpy()]
        y = g.loc[scored, "fp_won"].to_numpy()
        rows.append({"k": k, "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
                     "games": int(scored.sum())})
    table = pd.DataFrame(rows)
    return float(table.loc[table["log_loss"].idxmin(), "k"]), table


# --- history (everything strictly before a date) -----------------------------------------
@dataclass
class History:
    champ: pd.DataFrame     # champion_id -> champ_wr, presence, picks, wins, bans, window_games
    comfort: pd.DataFrame   # (team_id, champion_id) -> games, wins, role, all_games, comfort_games, comfort_edge
    team: pd.DataFrame      # team_id -> games, wins, wr (smoothed)
    window: list[str]       # patches in the champion window


def history_before(games: pd.DataFrame, actions: pd.DataFrame, before, cfg: FeatureConfig,
                   patch: str | None = None) -> History:
    before = pd.Timestamp(before)
    prior_g = games[games["date"] < before]
    prior_a = actions[actions["date"] < before]

    # Champion stats over all teams in the last N patches (up to `patch` if given).
    patches = sorted(prior_g["patch"].dropna().unique(), key=_patch_key)
    if patch is not None:
        patches = [p for p in patches if _patch_key(p) <= _patch_key(patch)]
    window = patches[-cfg.champ_patches:]
    n_w = int(prior_g["patch"].isin(window).sum())
    wa = prior_a[prior_a["patch"].isin(window)]
    picks = wa[wa["action"] == "pick"].groupby("champion_id")["won"].agg(picks="size", wins="sum")
    bans = wa[wa["action"] == "ban"].groupby("champion_id").size().rename("bans")
    champ = picks.join(bans, how="outer").fillna(0)
    champ["champ_wr"] = (champ["wins"] + cfg.champ_prior * 0.5) / (champ["picks"] + cfg.champ_prior) - 0.5
    champ["presence"] = (champ["picks"] + champ["bans"]) / n_w if n_w else 0.0
    champ["window_games"] = n_w

    # Team comfort: the team's own picks in the last `comfort_days` days.
    recent = prior_a[(prior_a["action"] == "pick")
                     & (prior_a["date"] >= before - pd.Timedelta(days=cfg.comfort_days))]
    team = (recent.drop_duplicates(["game_id", "team_id"]).groupby("team_id")["won"]
            .agg(games="size", wins="sum"))
    team["wr"] = (team["wins"] + cfg.comfort_prior * 0.5) / (team["games"] + cfg.comfort_prior)
    by_role = (recent.groupby(["team_id", "champion_id", "role"])["won"]
               .agg(games="size", wins="sum").reset_index())
    # Role-aware: count only games in the role the team usually plays this champion in.
    main = (by_role.sort_values("games", ascending=False, kind="stable")
            .drop_duplicates(["team_id", "champion_id"]).set_index(["team_id", "champion_id"]))
    main["all_games"] = recent.groupby(["team_id", "champion_id"]).size()
    twr = main.index.get_level_values("team_id").map(team["wr"]).to_numpy(dtype=float)
    main["comfort_games"] = np.log1p(main["games"])
    main["comfort_edge"] = (main["wins"] + cfg.comfort_prior * twr) / (main["games"] + cfg.comfort_prior) - twr
    return History(champ=champ, comfort=main, team=team, window=list(window))


def contributions(rows: pd.DataFrame, hist: History) -> pd.DataFrame:
    """Per-action feature values for `rows` (team_id, opponent_id, action, champion_id).
    Picks fill champ_wr/presence/comfort_*, bans fill denied; unknown champions add 0."""
    out = rows.copy()
    is_pick = (out["action"] == "pick").to_numpy()
    c = hist.champ.reindex(out["champion_id"])
    own = hist.comfort.reindex(pd.MultiIndex.from_arrays([out["team_id"], out["champion_id"]]))
    opp = hist.comfort.reindex(pd.MultiIndex.from_arrays([out["opponent_id"], out["champion_id"]]))
    for col, src in (("champ_wr", c["champ_wr"]), ("presence", c["presence"]),
                     ("comfort_games", own["comfort_games"]), ("comfort_edge", own["comfort_edge"])):
        out[col] = np.where(is_pick, src.fillna(0).to_numpy(), 0.0)
    out["denied"] = np.where(~is_pick, np.log1p(opp["all_games"].fillna(0).to_numpy()), 0.0)
    out["n_picks"] = is_pick.astype(float)
    out["n_bans"] = (~is_pick).astype(float)
    # sample sizes behind each number, for display
    out["champ_picks"] = c["picks"].fillna(0).to_numpy()
    out["champ_wins"] = c["wins"].fillna(0).to_numpy()
    out["window_games"] = int(hist.champ["window_games"].iloc[0]) if len(hist.champ) else 0
    out["team_games_on_champ"] = own["games"].fillna(0).to_numpy()
    out["team_wins_on_champ"] = own["wins"].fillna(0).to_numpy()
    out["team_role_on_champ"] = own["role"].to_numpy()
    return out


# --- feature table ------------------------------------------------------------------------
def build_features(drafts: pd.DataFrame, cfg: FeatureConfig = FeatureConfig(),
                   stages: list[int] = STAGES) -> pd.DataFrame:
    """One row per (game, stage). stage = draft actions completed (slots <= stage)."""
    games, actions = prepare(drafts)
    pre_fp, pre_sp, _ = elo(games, cfg.elo_k, cfg.elo_init)
    games = games.assign(elo_fp=pre_fp, elo_sp=pre_sp)

    parts = []
    for day, day_actions in actions.groupby("date", sort=True):
        parts.append(contributions(day_actions, history_before(games, actions, day, cfg)))
    contrib = pd.concat(parts, ignore_index=True)

    fp_idx = pd.MultiIndex.from_arrays([games.index, games["fp_team"]])
    sp_idx = pd.MultiIndex.from_arrays([games.index, games["sp_team"]])
    base = games[["date", "patch", "fp_team", "sp_team", "fp_blue", "fp_won"]].copy()
    base["elo_diff"] = games["elo_fp"] - games["elo_sp"]
    rows = []
    for s in stages:
        sums = contrib[contrib["draft_order"] <= s].groupby(["game_id", "team_id"])[DRAFT_COLS].sum()
        diff = sums.reindex(fp_idx).fillna(0).to_numpy() - sums.reindex(sp_idx).fillna(0).to_numpy()
        part = base.copy()
        part[[f"d_{c}" for c in DRAFT_COLS]] = diff
        part["stage"] = s
        rows.append(part)
    out = pd.concat(rows).rename_axis("game_id").reset_index()
    return out[["game_id", "stage", "date", "patch", "fp_team", "sp_team", "fp_won"] + FEATURES]


# --- models -------------------------------------------------------------------------------
def make_model(kind: str = "lr", C: float = 1.0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lr":
        return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    if kind == "gbm":
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                              min_samples_leaf=100, l2_regularization=1.0, random_state=0)
    raise ValueError(kind)


def fit(model, feats: pd.DataFrame, features: list[str]):
    """Fit on all rows of `feats`; each game's stages share one game's worth of weight."""
    w = 1.0 / feats.groupby("game_id")["stage"].transform("size")
    X, y = feats[features].to_numpy(), feats["fp_won"].to_numpy()
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w.to_numpy()})
    else:
        model.fit(X, y, sample_weight=w.to_numpy())
    return model


def predict(model, feats: pd.DataFrame, features: list[str]) -> np.ndarray:
    return model.predict_proba(feats[features].to_numpy())[:, 1]


def metrics(y, p) -> dict:
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"games": int(len(y)), "accuracy": float(np.mean((p >= 0.5) == y)),
            "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2)), "ece": calibration_error(y, p)}


def calibration_table(y, p, bins: int = 5) -> pd.DataFrame:
    """Quantile bins of predicted P(first-pick team wins): games, mean prediction, actual win rate."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p, dtype=float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    t = df.groupby("bin").agg(games=("y", "size"), p_low=("p", "min"), p_high=("p", "max"),
                              predicted=("p", "mean"), actual=("y", "mean"))
    return t.reset_index(drop=True)


def calibration_error(y, p, bins: int = 5) -> float:
    t = calibration_table(y, p, bins)
    return float((t["games"] * (t["predicted"] - t["actual"]).abs()).sum() / t["games"].sum())


def bootstrap_gain(y, p_base, p_main, n: int = 2000, seed: int = 0) -> dict:
    """Paired bootstrap over games: log-loss and accuracy gain of main over base, 95% CI."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pb, pm = np.clip(p_base, 1e-6, 1 - 1e-6), np.clip(p_main, 1e-6, 1 - 1e-6)
    ll_b = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    ll_m = -(y * np.log(pm) + (1 - y) * np.log(1 - pm))
    acc_b, acc_m = ((pb >= .5) == y).astype(float), ((pm >= .5) == y).astype(float)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    d_ll = (ll_b[idx] - ll_m[idx]).mean(axis=1)
    d_acc = (acc_m[idx] - acc_b[idx]).mean(axis=1)
    return {"log_loss_gain": float(ll_b.mean() - ll_m.mean()),
            "log_loss_gain_ci": [float(x) for x in np.percentile(d_ll, [2.5, 97.5])],
            "accuracy_gain": float(acc_m.mean() - acc_b.mean()),
            "accuracy_gain_ci": [float(x) for x in np.percentile(d_acc, [2.5, 97.5])]}


# --- training and validation --------------------------------------------------------------
def train_and_validate(drafts: pd.DataFrame | None = None, verbose: bool = True) -> dict:
    """Tune K on early data, choose the model on an inner validation split, then report
    once on the test period (games on/after TEST_START). Returns the report and the
    final model refitted on ALL games (for live use)."""
    log = print if verbose else (lambda *a, **k: None)
    df = load_drafts() if drafts is None else drafts
    games, _ = prepare(df)

    k, k_table = tune_elo_k(games)
    cfg = FeatureConfig(elo_k=k)
    log(f"Elo K = {k} (tuned on {int(k_table['games'].iloc[0])} games before {INNER_VAL_START})")
    feats = build_features(df, cfg)

    t_inner, t_test = pd.Timestamp(INNER_VAL_START), pd.Timestamp(TEST_START)
    inner_fit = feats[feats["date"] < t_inner]
    inner_val = feats[(feats["date"] >= t_inner) & (feats["date"] < t_test) & (feats["stage"] == FULL)]
    train = feats[feats["date"] < t_test]
    test = feats[feats["date"] >= t_test]
    test_full = test[test["stage"] == FULL]

    # 1. Model choice on the inner split only (test games untouched).
    candidates = [("lr", C) for C in LR_C_GRID] + [("gbm", None)]
    choice_rows = []
    for kind, C in candidates:
        m = fit(make_model(kind, C or 1.0), inner_fit, FEATURES)
        choice_rows.append({"model": kind, "C": C, **metrics(inner_val["fp_won"], predict(m, inner_val, FEATURES))})
    m = fit(make_model("lr"), inner_fit[inner_fit["stage"] == FULL], FEATURES)
    no_aug = metrics(inner_val["fp_won"], predict(m, inner_val, FEATURES))
    choice = pd.DataFrame(choice_rows)
    best = choice.loc[choice["log_loss"].idxmin()]
    kind, C = best["model"], (best["C"] if best["model"] == "lr" else 1.0)
    log(f"Chosen on inner validation ({len(inner_val)} games): {kind}" + (f", C={C}" if kind == "lr" else ""))

    # 2. Fit on all games before TEST_START, report on the test period.
    full_train = train[train["stage"] == FULL]
    models = {
        "Elo only": (fit(make_model("lr"), full_train, BASELINE_FEATURES), BASELINE_FEATURES),
        "Elo + side": (fit(make_model("lr"), full_train, SIDE_FEATURES), SIDE_FEATURES),
        "Elo + side + draft (main)": (fit(make_model(kind, C), train, FEATURES), FEATURES),
        "Gradient boosting (not chosen)" if kind == "lr" else "Logistic regression (not chosen)":
            (fit(make_model("gbm" if kind == "lr" else "lr"), train, FEATURES), FEATURES),
    }
    y = test_full["fp_won"].to_numpy()
    preds = {name: predict(m, test_full, f) for name, (m, f) in models.items()}
    raw_elo = elo_prob(test_full["elo_diff"])
    results = {"Raw Elo formula": metrics(y, raw_elo)} | {n: metrics(y, p) for n, p in preds.items()}

    main = models["Elo + side + draft (main)"][0]
    by_stage = []
    for s in (0, 6, 9, 12, 16, 18, 20):
        part = test[test["stage"] == s]
        by_stage.append({"stage": s, **metrics(part["fp_won"], predict(main, part, FEATURES))})

    g_test = games[games["date"] >= t_test]
    g_train = games[games["date"] < t_test]
    report = {
        "config": asdict(cfg), "elo_k_table": k_table.to_dict("records"),
        "model": {"kind": kind, "C": float(C) if kind == "lr" else None},
        "model_choice": choice.to_dict("records"), "inner_val_no_augmentation": no_aug,
        "split": {"train_games": int(len(g_train)), "test_games": int(len(g_test)),
                  "inner_fit_games": int(inner_fit["game_id"].nunique()),
                  "inner_val_games": int(len(inner_val)),
                  "train_dates": [str(g_train["date"].min().date()), str(g_train["date"].max().date())],
                  "test_dates": [str(g_test["date"].min().date()), str(g_test["date"].max().date())],
                  "test_events": sorted(g_test["event"].unique())},
        "test": results,
        "calibration": {"Elo only": calibration_table(y, preds["Elo only"]).to_dict("records"),
                        "main": calibration_table(y, preds["Elo + side + draft (main)"]).to_dict("records")},
        "draft_gain_over_elo_side": bootstrap_gain(y, preds["Elo + side"], preds["Elo + side + draft (main)"]),
        "draft_gain_over_elo": bootstrap_gain(y, preds["Elo only"], preds["Elo + side + draft (main)"]),
        "by_stage": by_stage,
        "sides": {
            "train_fp_win_rate": float(g_train["fp_won"].mean()), "test_fp_win_rate": float(g_test["fp_won"].mean()),
            "train_blue_win_rate": float((g_train["fp_won"] == g_train["fp_blue"]).mean()),
            "test_blue_win_rate": float((g_test["fp_won"] == g_test["fp_blue"]).mean()),
            "train_fp_on_red": int((g_train["fp_blue"] == 0).sum()),
            "test_fp_on_red": int((g_test["fp_blue"] == 0).sum()),
        },
    }
    if kind == "lr":
        lr = main.steps[-1][1]
        report["coefficients"] = dict(zip(FEATURES, map(float, lr.coef_[0])))

    # 3. Final model for live use: same choices, refitted on every game.
    final = fit(make_model(kind, C), feats, FEATURES)
    bundle = {"model": final, "kind": kind, "features": FEATURES, "config": asdict(cfg),
              "trained_through": str(games["date"].max().date()), "trained_games": int(len(games)),
              "report": report}
    return bundle


def save_model(bundle: dict, path: Path = MODEL_PATH) -> None:
    import joblib
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(rebuild: bool = False, path: Path = MODEL_PATH) -> dict:
    import joblib
    if rebuild or not path.exists():
        bundle = train_and_validate(verbose=False)
        save_model(bundle, path)
        return bundle
    return joblib.load(path)


# --- live / partial-draft prediction ------------------------------------------------------
def _resolve_team(x: str, drafts: pd.DataFrame) -> str:
    ids = set(drafts["team_id"].astype(str))
    if str(x) in ids:
        return str(x)
    names = drafts.sort_values("date").groupby("team_id")["team_name"].last().astype(str)
    hit = names[names.str.lower() == str(x).lower()]
    if hit.empty:
        raise ValueError(f"unknown team: {x!r}")
    return str(hit.index[0])


def _resolve_champions(xs, lookup: dict[str, str], what: str) -> list[str]:
    out = []
    for x in xs or ():
        key = str(x).lower()
        if key not in lookup:
            raise ValueError(f"unknown champion in {what}: {x!r}")
        out.append(lookup[key])
    return out


def predict_draft(team_a: str, team_b: str, side_a: str, picks_a=(), picks_b=(), bans=(),
                  date=None, patch: str | None = None, first_pick: str | None = None,
                  model: dict | None = None, drafts: pd.DataFrame | None = None) -> dict:
    """P(team_a wins) for a draft in progress (0-5 picks per side).

    team_a/team_b: team_id or team name. side_a: "blue" or "red".
    picks_a/picks_b: champion names or ids picked so far.
    bans: either a flat list (who banned what unknown: only checked for conflicts), or a
          dict {"a": [...], "b": [...]} of each team's bans, which also feeds `d_denied`.
    date: history cutoff, only games strictly before it are used (default: the day after
          the last game in the data). patch: optional, ends the champion window at it.
    first_pick: "a" or "b"; default is the blue-side team (always true before 2026).
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
        unknown_keys = set(bans) - {"a", "b"}
        if unknown_keys:
            raise ValueError(f"bans dict keys must be 'a'/'b', got {sorted(unknown_keys)}")
        ba, bb = _resolve_champions(bans.get("a"), lookup, "bans"), _resolve_champions(bans.get("b"), lookup, "bans")
        flat = ba + bb
    else:
        ba, bb, flat = [], [], _resolve_champions(bans, lookup, "bans")
    if len(pa) > 5 or len(pb) > 5:
        raise ValueError("at most 5 picks per side")
    if len(flat) > 10:
        raise ValueError("at most 10 bans")
    everything = pa + pb + flat
    if len(set(everything)) != len(everything):
        dup = sorted({names[c] for c in everything if everything.count(c) > 1})
        raise ValueError(f"champion used more than once in this draft: {dup}")

    games, actions = prepare(df)
    if date is None:
        cutoff = games["date"].max() + pd.Timedelta(days=1)
    else:
        cutoff = pd.Timestamp(date)
    prior = games[games["date"] < cutoff]
    _, _, ratings = elo(prior, cfg.elo_k, cfg.elo_init)
    hist = history_before(games, actions, cutoff, cfg, patch)

    a_first = (side_a == "blue") if first_pick is None else (first_pick == "a")
    rows = pd.DataFrame(
        [(a_id, b_id, "pick", c) for c in pa] + [(b_id, a_id, "pick", c) for c in pb]
        + [(a_id, b_id, "ban", c) for c in ba] + [(b_id, a_id, "ban", c) for c in bb],
        columns=["team_id", "opponent_id", "action", "champion_id"])
    contrib = contributions(rows, hist)
    sums = contrib.groupby("team_id")[DRAFT_COLS].sum().reindex([a_id, b_id]).fillna(0)
    diff_a = sums.loc[a_id] - sums.loc[b_id]

    elo_a, elo_b = ratings.get(a_id, cfg.elo_init), ratings.get(b_id, cfg.elo_init)
    sign = 1 if a_first else -1          # features are from the first-pick team's side
    a_blue = side_a == "blue"
    x = {"elo_diff": sign * (elo_a - elo_b), "fp_blue": int(a_blue == a_first)}
    x |= {f"d_{c}": sign * float(diff_a[c]) for c in DRAFT_COLS}
    X = pd.DataFrame([x])[bundle["features"]].to_numpy()
    p_fp = float(bundle["model"].predict_proba(X)[0, 1])
    p_a = p_fp if a_first else 1 - p_fp

    out = {
        "team_a": a_id, "team_b": b_id, "first_pick": a_id if a_first else b_id,
        "p_team_a": p_a, "p_team_b": 1 - p_a,
        "p_team_a_elo_only": float(elo_prob(elo_a - elo_b)),
        "elo": {"team_a": elo_a, "team_b": elo_b},
        "features_first_pick_view": x,
        "history_before": str(cutoff.date()), "history_games": int(len(prior)),
        "champion_window": hist.window,
        "team_games": {t: int(hist.team["games"].get(t, 0)) for t in (a_id, b_id)},
        "actions": contrib.assign(champion=contrib["champion_id"].map(names),
                                  side=np.where(contrib["team_id"] == a_id, "a", "b")),
    }
    if bundle["kind"] == "lr":   # per-feature push on the log-odds, from team_a's view
        scaler, lr = bundle["model"].steps[0][1], bundle["model"].steps[-1][1]
        z = (X[0] - scaler.mean_) / scaler.scale_ * lr.coef_[0]
        out["contributions"] = {f: float(v) * (1 if a_first else -1) for f, v in zip(bundle["features"], z)}
    return out


# --- CLI ----------------------------------------------------------------------------------
def _fmt_report(r: dict) -> str:
    lines = []
    s = r["split"]
    lines.append(f"Train: {s['train_games']} games {s['train_dates'][0]}..{s['train_dates'][1]};  "
                 f"test: {s['test_games']} games {s['test_dates'][0]}..{s['test_dates'][1]}")
    lines.append(f"Model: {r['model']}")
    lines.append("\n| Model | Games | Accuracy | Log loss | Brier | Calib. error |\n|---|---|---|---|---|---|")
    for name, m in r["test"].items():
        lines.append(f"| {name} | {m['games']} | {m['accuracy']:.1%} | {m['log_loss']:.4f} | {m['brier']:.4f} | {m['ece']:.3f} |")
    for key, label in (("draft_gain_over_elo_side", "Elo + side"), ("draft_gain_over_elo", "Elo only")):
        g = r[key]
        lines.append(f"\nDraft gain over {label}: log loss {g['log_loss_gain']:+.4f} "
                     f"(95% CI {g['log_loss_gain_ci'][0]:+.4f}..{g['log_loss_gain_ci'][1]:+.4f}); "
                     f"accuracy {g['accuracy_gain']:+.1%} (95% CI {g['accuracy_gain_ci'][0]:+.1%}..{g['accuracy_gain_ci'][1]:+.1%})")
    for name, rows in r["calibration"].items():
        lines.append(f"\nCalibration, {name}:\n| Predicted range | Games | Mean predicted | Actual |\n|---|---|---|---|")
        for row in rows:
            lines.append(f"| {row['p_low']:.0%}-{row['p_high']:.0%} | {row['games']} | {row['predicted']:.1%} | {row['actual']:.1%} |")
    lines.append("\nBy draft stage (test, main model):\n| Actions done | Accuracy | Log loss | Brier |\n|---|---|---|---|")
    for row in r["by_stage"]:
        lines.append(f"| {row['stage']} | {row['accuracy']:.1%} | {row['log_loss']:.4f} | {row['brier']:.4f} |")
    lines.append("\nModel choice (inner validation):")
    lines.append(pd.DataFrame(r["model_choice"]).to_string(index=False))
    lines.append(f"LR without stage augmentation: {r['inner_val_no_augmentation']}")
    lines.append("\nElo K:\n" + pd.DataFrame(r["elo_k_table"]).to_string(index=False))
    lines.append(f"\nSides: {r['sides']}")
    if "coefficients" in r:
        lines.append("Coefficients (standardised): " + ", ".join(f"{k} {v:+.3f}" for k, v in r["coefficients"].items()))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    bundle = train_and_validate()
    print(_fmt_report(bundle["report"]))
    save_model(bundle)
    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
