"""Draft simulator: a 20-slot pro draft, scored step by step as a decision aid.

The win model (model.py) shifts odds by only a few points per draft (docs/MODEL.md),
so everything here is framed as "what each choice shifts", with sample sizes and
an uncertainty band, never as a forecast of the winner.

- Board: the real pro order (bans 1-6, picks 1-6, bans 7-10, picks 7-10), keyed on
  which team picks first; our side is chosen separately (2026 first selection).
- Win chance: calibrated model estimate for our team, with a 90% band from 50
  bootstrap refits of the model (resampled series), plus team strength (Elo) alone.
- Best options (our slot):
    pick: every plausible available champion, scored by how much it moves our win
          chance, with the main reason (player comfort / team comfort / lane matchup /
          meta strength) and the sample sizes behind it.
    ban:  value = P(they pick it later this draft) x (how much worse it is for us than
          their typical pick). Bans do NOT enter the win model directly: its ban feature
          is an association (underdogs ban the favourite's comfort picks), not an effect.
- Likely from them: the opponent's next pick (or ban) distribution from their recent
  tendencies by draft phase, the league meta, their open roles and what is available.
  Fixed weights, checked walk-forward by `likely_accuracy`.
Hard Fearless: champions used earlier in the series are locked for both teams.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer import model as M  # noqa: E402
from draft_analyzer.analysis import PHASE1_PICKS, SMALL_SAMPLE, _patch_key  # noqa: E402
from draft_analyzer.config import CACHE_DIR  # noqa: E402
from draft_analyzer.drafts import DRAFT_TEMPLATE  # noqa: E402

ROLES = M.ROLES
N_SLOTS = len(DRAFT_TEMPLATE)

# "Likely from them": fixed in advance, checked walk-forward (likely_accuracy).
TENDENCY_DAYS = 365          # look-back for team tendencies and role habits
TENDENCY_HALF_LIFE = 3.0     # patches until a game counts half
TEAM_WEIGHT = 0.7            # team tendency vs league meta for picks
OTHER_PHASE_WEIGHT = 0.3     # a champion picked in the other pick phase still counts this much
BAN_WEIGHTS = {"team": 0.5, "meta": 0.25, "target": 0.25}   # their ban habits / league / what we play
MIN_ROLE_GAMES = 2           # a champion is a candidate for a role only with this many pro games there

REASONS = {   # model features behind each reason, as shown to coaches
    "player comfort": ["d_player_games", "d_player_edge"],
    "team comfort": ["d_team_games", "d_team_edge"],
    "lane matchup": ["matchup"],
    "meta strength": ["d_champ_wr", "d_presence", "d_priority"],
}
BAND = (5, 95)               # percentiles of the bootstrap refits shown as the uncertainty band
LIKELY_EVAL_PATH = CACHE_DIR / "likely_eval.joblib"


# --- board --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Setup:
    """Everything about the game except the draft actions. Hashable, for caching."""
    our_side: str = "blue"
    we_first: bool = True                  # do we make the first pick (slot 7)?
    game_number: int = 1                   # in the series; Fearless locks come from earlier games
    locked: frozenset = frozenset()        # champion_ids used earlier in the series (hard Fearless)
    lineup_us: tuple = ()                  # ((role, player_id), ...); missing roles use the last lineup
    lineup_them: tuple = ()

    def __post_init__(self):
        if self.our_side not in ("blue", "red"):
            raise ValueError("our_side must be 'blue' or 'red'")
        if not 1 <= self.game_number <= 5:
            raise ValueError("game_number must be 1-5")


@dataclass(frozen=True)
class Action:
    slot: int                 # 1-20
    champion_id: str
    role: str | None = None   # picks only


def slots(setup: Setup) -> list[dict]:
    """The 20 draft slots in order: who acts (us/them), pick or ban, phase and a label."""
    out, n = [], {"ban": 0, "pick": 0}
    for i, (fs, kind) in enumerate(DRAFT_TEMPLATE, 1):
        who = "us" if (fs == "F") == setup.we_first else "them"
        n[kind] += 1
        phase = (1 if i in PHASE1_PICKS else 2) if kind == "pick" else (1 if i <= 6 else 2)
        side = setup.our_side if who == "us" else ("red" if setup.our_side == "blue" else "blue")
        out.append({"slot": i, "who": who, "kind": kind, "phase": phase, "side": side,
                    "label": f"{kind.capitalize()} {n[kind]}"})
    return out


# --- opponent tendencies (for "likely from them") ----------------------------------------
@dataclass
class Tendencies:
    """Pick/ban habits of every team over the TENDENCY_DAYS before a date, as dense arrays
    over `champs`. Recency-weighted by patch; raw counts kept for the evidence shown."""
    champs: pd.Index
    picks: pd.DataFrame            # team_id, ci, phase, w, role
    bans: pd.DataFrame             # team_id, ci, phase, w
    games: pd.DataFrame            # team_id -> games (raw), w_games
    meta_pick: dict                # phase -> share array
    meta_ban: dict
    role_global: np.ndarray        # champs x 5 pick counts
    _team: dict = field(default_factory=dict)

    def team(self, team_id: str) -> dict:
        if team_id not in self._team:
            n = len(self.champs)
            p, b = self.picks[self.picks["team_id"] == team_id], self.bans[self.bans["team_id"] == team_id]
            t = {}
            for ph in (1, 2):
                pp, bb = p[p["phase"] == ph], b[b["phase"] == ph]
                t[f"pick_w{ph}"] = np.bincount(pp["ci"], weights=pp["w"], minlength=n)
                t[f"pick_n{ph}"] = np.bincount(pp["ci"], minlength=n)
                t[f"ban_w{ph}"] = np.bincount(bb["ci"], weights=bb["w"], minlength=n)
                t[f"ban_n{ph}"] = np.bincount(bb["ci"], minlength=n)
            roles = np.zeros((n, len(ROLES)))
            known = p[p["role"].isin(ROLES)]
            np.add.at(roles, (known["ci"].to_numpy(dtype=int), known["role"].map(ROLES.index).to_numpy(dtype=int)), 1)
            t["roles"] = roles
            g = self.games.reindex([team_id]).fillna(0).iloc[0]
            t["games"], t["w_games"] = int(g["games"]), float(g["w_games"])
            self._team[team_id] = t
        return self._team[team_id]

    def role_probs(self, team_id: str) -> np.ndarray:
        """P(role | champion) for this team: its own habits plus the league's (champs x 5)."""
        g = self.role_global
        frac = np.divide(g, g.sum(axis=1, keepdims=True), out=np.full_like(g, 0.2), where=g.sum(axis=1, keepdims=True) > 0)
        r = self.team(team_id)["roles"] + 2 * frac + 0.02
        return r / r.sum(axis=1, keepdims=True)


def tendency_tables(actions: pd.DataFrame, before, champs: pd.Index) -> Tendencies:
    """Habits from games strictly before `before` (and within TENDENCY_DAYS)."""
    before = pd.Timestamp(before)
    a = actions[(actions["date"] < before) & (actions["date"] >= before - pd.Timedelta(days=TENDENCY_DAYS))]
    a = a[a["champion_id"].isin(champs)].copy()
    patches = sorted(a["patch"].dropna().unique(), key=_patch_key)
    pidx = {p: i for i, p in enumerate(patches)}
    a["w"] = 0.5 ** ((len(patches) - 1 - a["patch"].map(pidx).fillna(0)) / TENDENCY_HALF_LIFE)
    a["ci"] = a["champion_id"].map({c: i for i, c in enumerate(champs)}).astype(int)
    pick = a["action"] == "pick"
    a["phase"] = np.where(pick, np.where(a["draft_order"].isin(PHASE1_PICKS), 1, 2),
                          np.where(a["draft_order"] <= 6, 1, 2))
    picks, bans = a[pick], a[~pick]
    n = len(champs)

    def share(df):
        c = np.bincount(df["ci"], minlength=n).astype(float)
        return c / c.sum() if c.sum() else np.full(n, 1 / n)

    role_global = np.zeros((n, len(ROLES)))
    known = picks[picks["role"].isin(ROLES)]
    np.add.at(role_global, (known["ci"].to_numpy(dtype=int), known["role"].map(ROLES.index).to_numpy(dtype=int)), 1)
    games = (a.drop_duplicates(["game_id", "team_id"]).groupby("team_id")["w"]
             .agg(games="size", w_games="sum"))
    return Tendencies(champs=champs, picks=picks[["team_id", "ci", "phase", "w", "role"]],
                      bans=bans[["team_id", "ci", "phase", "w"]], games=games,
                      meta_pick={ph: share(picks[picks["phase"] == ph]) for ph in (1, 2)},
                      meta_ban={ph: share(bans[bans["phase"] == ph]) for ph in (1, 2)},
                      role_global=role_global)


def pick_probs(tend: Tendencies, team_id: str, phase: int, open_roles: list[str], available: np.ndarray,
               team_weight: float = TEAM_WEIGHT) -> np.ndarray:
    """P(team picks each champion next), over available champions that fit an open role."""
    t = tend.team(team_id)
    tw = t[f"pick_w{phase}"] + OTHER_PHASE_WEIGHT * t[f"pick_w{3 - phase}"]
    lam = team_weight if tw.sum() > 0 else 0.0
    team_share = tw / tw.sum() if tw.sum() > 0 else 0.0
    base = lam * team_share + (1 - lam) * tend.meta_pick[phase] + 1e-4
    fit = tend.role_probs(team_id)[:, [ROLES.index(r) for r in open_roles]].sum(axis=1) if open_roles else 0.0
    p = base * fit * available
    return p / p.sum() if p.sum() > 0 else p


def ban_probs(tend: Tendencies, team_id: str, target_id: str, phase: int, available: np.ndarray,
              weights: dict = BAN_WEIGHTS) -> np.ndarray:
    """P(team bans each champion next): its ban habits in this phase, the league's, and
    what the target team plays."""
    t, tt = tend.team(team_id), tend.team(target_id)
    parts = {"team": t[f"ban_w{phase}"], "meta": tend.meta_ban[phase],
             "target": tt["pick_w1"] + tt["pick_w2"]}
    p = np.full(len(tend.champs), 1e-4)
    used = {k: v for k, v in parts.items() if v.sum() > 0}
    total_w = sum(weights[k] for k in used)
    for k, v in used.items():
        p = p + weights[k] / total_w * v / v.sum()
    p = p * available
    return p / p.sum() if p.sum() > 0 else p


# --- the simulator ------------------------------------------------------------------------
class Simulator:
    """Scores drafts between `our_id` and `opp_id`. Build once per matchup (it holds the
    history, ratings and tendencies); every method then takes a Setup and the actions
    so far and is fast enough to call on every click."""

    def __init__(self, our_id: str, opp_id: str, live: M.Live):
        self.live, self.us, self.them = live, our_id, opp_id
        self.champs = live.names.index
        self.cidx = {c: i for i, c in enumerate(self.champs)}
        self.tend = tendency_tables(live.actions, live.cutoff, self.champs)
        self.F = list(live.bundle["features"])
        self.fi = {f: i for i, f in enumerate(self.F)}
        self._xcache: dict = {}        # (setup, actions) -> features: each click adds one prefix
        self._lineups: dict = {}

    # --- people ---------------------------------------------------------------------------
    def team_id(self, who: str) -> str:
        return self.us if who == "us" else self.them

    @cached_property
    def player_names(self) -> pd.Series:
        d = self.live.drafts.dropna(subset=["player_id"]).sort_values("date")
        return d.groupby("player_id")["player_name"].last().astype(str).str.strip()

    def roster(self, who: str) -> pd.DataFrame:
        """Players who played for the team in the tendency window: id, name, main role, games."""
        a = self.live.actions
        a = a[(a["team_id"] == self.team_id(who)) & (a["action"] == "pick") & a["player_id"].notna()
              & (a["date"] >= self.live.cutoff - pd.Timedelta(days=TENDENCY_DAYS))]
        if a.empty:
            return pd.DataFrame(columns=["player_id", "name", "role", "games", "last_played"])
        out = a.groupby("player_id").agg(games=("game_id", "nunique"), last_played=("date", "max"),
                                         role=("role", lambda r: r.value_counts().index[0]))
        out["name"] = self.player_names.reindex(out.index).fillna(out.index.to_series())
        return out.reset_index().sort_values(["role", "last_played"], ascending=[True, False])

    def default_lineup(self, who: str) -> dict[str, str]:
        """role -> player_id from the team's most recent game."""
        if who not in self._lineups:
            self._lineups[who] = M.latest_lineup(self.live.actions, self.team_id(who), self.live.cutoff)
        return dict(self._lineups[who])

    def lineup(self, setup: Setup, who: str) -> dict[str, str]:
        return self.default_lineup(who) | dict(setup.lineup_us if who == "us" else setup.lineup_them)

    # --- board state ----------------------------------------------------------------------
    def unavailable(self, setup: Setup, actions) -> set[str]:
        return set(setup.locked) | {a.champion_id for a in actions}

    def available_mask(self, setup: Setup, actions) -> np.ndarray:
        gone = self.unavailable(setup, actions)
        return np.array([c not in gone for c in self.champs], dtype=float)

    def picks_of(self, setup: Setup, actions, who: str) -> list[Action]:
        s = slots(setup)
        return [a for a in actions if s[a.slot - 1]["who"] == who and s[a.slot - 1]["kind"] == "pick"]

    def open_roles(self, setup: Setup, actions, who: str) -> list[str]:
        taken = {a.role for a in self.picks_of(setup, actions, who)}
        return [r for r in ROLES if r not in taken]

    def guess_role(self, setup: Setup, actions, who: str, champion_id: str) -> str:
        open_r = self.open_roles(setup, actions, who) or ROLES
        rp = self.tend.role_probs(self.team_id(who))[self.cidx[champion_id]]
        return max(open_r, key=lambda r: rp[ROLES.index(r)])

    def apply(self, setup: Setup, actions, champion_id: str, role: str | None = None) -> tuple:
        """The draft with the next slot filled. Raises ValueError for an illegal action."""
        actions = tuple(actions)
        if len(actions) >= N_SLOTS:
            raise ValueError("the draft is complete")
        if champion_id not in self.cidx:
            raise ValueError(f"unknown champion: {champion_id!r}")
        if champion_id in self.unavailable(setup, actions):
            why = "locked by Fearless" if champion_id in setup.locked else "already picked or banned"
            raise ValueError(f"{self.live.names[champion_id]} is {why}")
        slot = slots(setup)[len(actions)]
        if slot["kind"] == "ban":
            role = None
        else:
            open_r = self.open_roles(setup, actions, slot["who"])
            role = role or self.guess_role(setup, actions, slot["who"], champion_id)
            if role not in open_r:
                raise ValueError(f"role {role!r} is already filled for this team")
        return actions + (Action(slot["slot"], champion_id, role),)

    def set_role(self, setup: Setup, actions, slot: int, role: str) -> tuple:
        """Change a pick's role; if a teammate already has that role, the two swap."""
        actions = list(actions)
        i = next(i for i, a in enumerate(actions) if a.slot == slot)
        who = slots(setup)[slot - 1]["who"]
        for j, a in enumerate(actions):
            if j != i and a.role == role and slots(setup)[a.slot - 1]["who"] == who:
                actions[j] = replace(a, role=actions[i].role)
        actions[i] = replace(actions[i], role=role)
        return tuple(actions)

    # --- win chance -----------------------------------------------------------------------
    def _side_ids(self, setup: Setup) -> tuple[str, str]:
        return (self.us, self.them) if setup.our_side == "blue" else (self.them, self.us)

    def _rows(self, setup: Setup, actions) -> pd.DataFrame:
        """Model rows for the picks so far (bans are left out of the win model, see module doc)."""
        s, lineups = slots(setup), {w: self.lineup(setup, w) for w in ("us", "them")}
        rows = []
        for a in actions:
            sl = s[a.slot - 1]
            if sl["kind"] == "pick":
                who = sl["who"]
                rows.append((self.team_id(who), self.team_id("them" if who == "us" else "us"), sl["side"],
                             "pick", a.champion_id, a.role, lineups[who].get(a.role)))
        return pd.DataFrame(rows, columns=["team_id", "opponent_id", "side", "action", "champion_id", "role",
                                           "player_id"])

    def _x(self, setup: Setup, actions) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
        key = (setup, tuple(actions))
        if key not in self._xcache:
            if len(self._xcache) > 5000:
                self._xcache.clear()
            blue, red = self._side_ids(setup)
            x, contrib, mu = M.draft_x(self.live, blue, red, self._rows(setup, actions),
                                       int(setup.we_first == (setup.our_side == "blue")))
            self._xcache[key] = (np.array([x[f] for f in self.F]), contrib, mu)
        return self._xcache[key]

    def _ours(self, p_blue: np.ndarray, setup: Setup) -> np.ndarray:
        return p_blue if setup.our_side == "blue" else 1 - p_blue

    def win_chance(self, setup: Setup, actions) -> dict:
        X, _, _ = self._x(setup, actions)
        p = float(self._ours(M.predict_x(self.live.bundle, X), setup)[0])
        boot = self._ours(M.predict_boot(self.live.bundle, X)[:, 0], setup)
        lo, hi = np.percentile(boot, BAND)
        r = self.live.ratings
        elo_us, elo_them = r.get(self.us, self.live.cfg.elo_init), r.get(self.them, self.live.cfg.elo_init)
        return {"p": p, "lo": float(lo), "hi": float(hi), "p_elo": float(M.elo_prob(elo_us - elo_them)),
                "elo_us": elo_us, "elo_them": elo_them, "refits": len(self.live.bundle["bootstrap"])}

    def trajectory(self, setup: Setup, actions) -> pd.DataFrame:
        """Win chance after 0, 1, ..., len(actions) actions (one batch prediction)."""
        actions = tuple(actions)
        X = np.stack([self._x(setup, actions[:k])[0] for k in range(len(actions) + 1)])
        p = self._ours(M.predict_x(self.live.bundle, X), setup)
        boot = self._ours(M.predict_boot(self.live.bundle, X), setup)
        lo, hi = np.percentile(boot, BAND, axis=0)
        s = slots(setup)
        labels = ["Start"] + [f"{s[a.slot - 1]['label']} ({'us' if s[a.slot - 1]['who'] == 'us' else 'them'}): "
                              f"{self.live.names[a.champion_id]}" for a in actions]
        short = ["Start"] + [s[a.slot - 1]["label"].replace("Ban ", "B").replace("Pick ", "P") for a in actions]
        return pd.DataFrame({"step": range(len(actions) + 1), "short": short, "label": labels,
                             "p": p, "lo": lo, "hi": hi})

    # --- candidates -----------------------------------------------------------------------
    def _candidates(self, setup: Setup, actions, who: str, available: np.ndarray) -> pd.DataFrame:
        """Available champions that fit one of the team's open roles, with that role."""
        open_r = self.open_roles(setup, actions, who)
        if not open_r:
            return pd.DataFrame(columns=["champion_id", "role"])
        cols = [ROLES.index(r) for r in open_r]
        team_roles = self.tend.team(self.team_id(who))["roles"][:, cols]
        ok = (self.tend.role_global[:, cols] >= MIN_ROLE_GAMES) | (team_roles >= 1)
        rp = self.tend.role_probs(self.team_id(who))[:, cols] * ok
        keep = (rp.sum(axis=1) > 0) & (available > 0)
        idx = np.flatnonzero(keep)
        return pd.DataFrame({"champion_id": self.champs[idx], "role": [open_r[k] for k in rp[idx].argmax(axis=1)]})

    def _pick_effects(self, setup: Setup, actions, who: str, cand: pd.DataFrame, X_now: np.ndarray):
        """Feature rows after `who` picks each candidate, plus per-candidate evidence."""
        s_side = setup.our_side if who == "us" else ("red" if setup.our_side == "blue" else "blue")
        other = "them" if who == "us" else "us"
        lineup = self.lineup(setup, who)
        rows = cand.assign(team_id=self.team_id(who), opponent_id=self.team_id(other), side=s_side,
                           action="pick", player_id=cand["role"].map(lineup), game_id="cand", draft_order=0)
        contrib = M.contributions(rows, self.live.hist)
        sign = 1.0 if s_side == "blue" else -1.0
        D = np.zeros((len(cand), len(self.F)))
        for c in M.DRAFT_COLS:
            D[:, self.fi[f"d_{c}"]] = sign * contrib[c].to_numpy()
        # Lane matchup against the other team's pick in the same role, if made.
        opp_by_role = {a.role: a.champion_id for a in self.picks_of(setup, actions, other)}
        opp = cand["role"].map(opp_by_role).astype(object)
        has = opp.notna().to_numpy()
        m = self.live.hist.matchup
        ours = m.reindex(pd.MultiIndex.from_arrays([cand["role"], cand["champion_id"], opp.where(opp.notna(), "")]))
        edge = ours["edge"].fillna(0).to_numpy()          # from `who`'s side
        D[:, self.fi["matchup"]] = np.where(has, sign * edge, 0.0)
        contrib["matchup_vs"] = opp.map(self.live.names).to_numpy()
        contrib["matchup_games"] = np.where(has, ours["games"].fillna(0).to_numpy(), 0)
        contrib["matchup_wins"] = np.where(has, ours["wins"].fillna(0).to_numpy(), 0)
        return X_now + D, contrib

    def _reasons(self, X_new: np.ndarray, X_now: np.ndarray, toward_us: float) -> pd.DataFrame:
        """Push of each reason group on our log-odds (toward_us: +1 if blue is us)."""
        push = M.feature_push(self.live.bundle, X_new, X_now) * toward_us
        return pd.DataFrame({k: push[:, [self.fi[f] for f in fs]].sum(axis=1) for k, fs in REASONS.items()})

    def _evidence(self, r, who: str) -> dict[str, str]:
        """Plain-English evidence per reason, with sample sizes."""
        def rec(w, g):
            return f"{int(w)}–{int(g - w)}" + (" · small sample" if g < SMALL_SAMPLE else "")
        team_name = "We" if who == "us" else "They"
        pname = self.player_names.get(r.player_id, "the player") if isinstance(r.player_id, str) else "the player"
        out = {
            "player comfort": (f"{pname}: {int(r.player_games_n)} games on it ({rec(r.player_wins_n, r.player_games_n)})"
                               if r.player_games_n else f"{pname}: no games on it in the last year"),
            "team comfort": (f"{team_name} played it as {r.role} {int(r.team_games_n)}× ({rec(r.team_wins_n, r.team_games_n)})"
                             if r.team_games_n else f"{team_name} haven't played it as {r.role} in the last year"),
            "meta strength": f"League: {rec(r.champ_wins, r.champ_picks)} in the last 3 patches "
                             f"({int(r.window_games)} games)",
        }
        if r.matchup_vs is not None and isinstance(r.matchup_vs, str):
            out["lane matchup"] = (f"vs {r.matchup_vs} ({r.role}): {rec(r.matchup_wins, r.matchup_games)} in pro games"
                                   if r.matchup_games else f"vs {r.matchup_vs} ({r.role}): no pro games, counted as 50%")
        else:
            out["lane matchup"] = "no lane opponent picked yet"
        return out

    def best_options(self, setup: Setup, actions, n: int = 5) -> pd.DataFrame | None:
        """Top `n` candidates for our current slot, or None if it isn't our turn."""
        actions = tuple(actions)
        if len(actions) >= N_SLOTS:
            return None
        slot = slots(setup)[len(actions)]
        if slot["who"] != "us":
            return None
        X_now, _, _ = self._x(setup, actions)
        bundle, avail = self.live.bundle, self.available_mask(setup, actions)
        p_now = self._ours(M.predict_x(bundle, X_now), setup)[0]
        boot_now = self._ours(M.predict_boot(bundle, X_now)[:, 0], setup)
        us_blue = 1.0 if setup.our_side == "blue" else -1.0
        if slot["kind"] == "pick":
            cand = self._candidates(setup, actions, "us", avail)
            if cand.empty:
                return cand
            Xc, contrib = self._pick_effects(setup, actions, "us", cand, X_now)
            shift = self._ours(M.predict_x(bundle, Xc), setup) - p_now
            boot = self._ours(M.predict_boot(bundle, Xc), setup) - boot_now[:, None]
            reasons = self._reasons(Xc, X_now, us_blue)
            out = contrib.assign(shift=shift, lo=np.percentile(boot, BAND[0], axis=0),
                                 hi=np.percentile(boot, BAND[1], axis=0))
            out = pd.concat([out.reset_index(drop=True), reasons], axis=1)
            out = out.sort_values("shift", ascending=False).head(n)
            out["p_them"] = np.nan
            return self._finish(out, "us")

        # Ban: value = P(they pick it later) x (how much worse for us than their typical pick).
        their_slots = [s for s in slots(setup)[len(actions):] if s["who"] == "them" and s["kind"] == "pick"]
        if not their_slots:
            return pd.DataFrame()
        open_them = self.open_roles(setup, actions, "them")
        p_next = pick_probs(self.tend, self.them, their_slots[0]["phase"], open_them, avail)
        p_any = np.minimum(1.0, len(their_slots) * p_next)
        cand = self._candidates(setup, actions, "them", avail)
        cand = cand[p_next[[self.cidx[c] for c in cand["champion_id"]]] > 0].reset_index(drop=True)
        if cand.empty:
            return cand
        ci = np.array([self.cidx[c] for c in cand["champion_id"]])
        Xc, contrib = self._pick_effects(setup, actions, "them", cand, X_now)
        shift = self._ours(M.predict_x(bundle, Xc), setup) - p_now           # <0: good for them
        boot = self._ours(M.predict_boot(bundle, Xc), setup) - boot_now[:, None]
        w = p_next[ci] / p_next[ci].sum()
        typical, typical_boot = (w * shift).sum(), (boot * w).sum(axis=1, keepdims=True)
        value = p_any[ci] * (typical - shift)
        value_boot = p_any[ci] * (typical_boot - boot)
        reasons = -self._reasons(Xc, X_now, us_blue)      # what their pick would gain them
        out = contrib.assign(shift=value, lo=np.percentile(value_boot, BAND[0], axis=0),
                             hi=np.percentile(value_boot, BAND[1], axis=0), p_them=p_any[ci],
                             their_shift=shift)
        out = pd.concat([out.reset_index(drop=True), reasons], axis=1)
        out = out.sort_values("shift", ascending=False).head(n)
        return self._finish(out, "them")

    def _finish(self, out: pd.DataFrame, evidence_for: str) -> pd.DataFrame:
        groups = list(REASONS)
        main = out[groups].idxmax(axis=1).where(out[groups].max(axis=1) > 0, "no single strong reason")
        ev = [self._evidence(r, evidence_for) for r in out.itertuples()]
        out = out.assign(champion=out["champion_id"].map(self.live.names), reason=main,
                         evidence=[e.get(m, "") for e, m in zip(ev, main)],
                         all_evidence=[" · ".join(e.values()) for e in ev])
        if evidence_for == "them":
            label = {"player comfort": "denies their comfort pick", "team comfort": "denies their comfort pick",
                     "lane matchup": "denies a lane matchup they win", "meta strength": "denies a strong meta pick"}
            out["reason"] = out["reason"].map(label).fillna("no single strong reason")
            out["evidence"] = [f"They pick it {p:.0%} of the time from here · {e}" for p, e in zip(out["p_them"], out["evidence"])]
        out["shift_pts"], out["lo_pts"], out["hi_pts"] = out["shift"] * 100, out["lo"] * 100, out["hi"] * 100
        return out[["champion_id", "champion", "role", "shift_pts", "lo_pts", "hi_pts", "reason", "evidence",
                    "all_evidence", "p_them"]].reset_index(drop=True)

    # --- likely from them -----------------------------------------------------------------
    def likely_next(self, setup: Setup, actions, n: int = 5) -> dict | None:
        """The opponent's most likely choices for their next slot (now or later)."""
        actions = tuple(actions)
        nxt = next((s for s in slots(setup)[len(actions):] if s["who"] == "them"), None)
        if nxt is None:
            return None
        avail = self.available_mask(setup, actions)
        t, tu = self.tend.team(self.them), self.tend.team(self.us)
        if nxt["kind"] == "pick":
            open_r = self.open_roles(setup, actions, "them")
            p = pick_probs(self.tend, self.them, nxt["phase"], open_r, avail)
        else:
            p = ban_probs(self.tend, self.them, self.us, nxt["phase"], avail)
        top = np.argsort(-p)[:n]
        rows = []
        champ_meta = self.live.hist.champ
        for i in top:
            c = self.champs[i]
            meta = champ_meta.loc[c] if c in champ_meta.index else None
            meta_txt = (f"league: in {meta['presence']:.0%} of games (last 3 patches, {int(meta['window_games'])} games)"
                        if meta is not None else "league: not played in the last 3 patches")
            if nxt["kind"] == "pick":
                n1, n2 = int(t["pick_n1"][i]), int(t["pick_n2"][i])
                role = max(open_r, key=lambda r: self.tend.role_probs(self.them)[i, ROLES.index(r)]) if open_r else None
                ev = f"Picked {n1 + n2}× in their last {t['games']} games ({n1} in phase 1) · {meta_txt}"
                small = (n1 + n2) < SMALL_SAMPLE
            else:
                b = int(t[f"ban_n{nxt['phase']}"][i])
                ours = int(tu["pick_n1"][i] + tu["pick_n2"][i])
                role = None
                ev = (f"Banned {b}× in phase {nxt['phase']} of their last {t['games']} games · "
                      f"we picked it {ours}× in our last {tu['games']} games · {meta_txt}")
                small = b < SMALL_SAMPLE
            rows.append({"champion_id": c, "champion": self.live.names[c], "role": role, "prob": float(p[i]),
                         "evidence": ev, "small_sample": small})
        return {"slot": nxt, "kind": nxt["kind"], "table": pd.DataFrame(rows)}


# --- checking "likely from them" on past games ---------------------------------------------
def likely_accuracy(live: M.Live, start: str = "2025-07-01") -> pd.DataFrame:
    """Walk-forward check of pick_probs / ban_probs: for every pick and ban in games from
    `start`, using only games on earlier dates, was the actual champion the top guess or
    in the top 5? Compared with the league meta alone. Hard Fearless locks applied."""
    games, actions, champs = live.games, live.actions, live.names.index
    cidx = {c: i for i, c in enumerate(champs)}
    a = actions[actions["champion_id"].isin(cidx)].copy()
    a["ci"] = a["champion_id"].map(cidx)
    a = a.merge(games[["series_id", "game_number"]], left_on="game_id", right_index=True)
    test = a[a["date"] >= pd.Timestamp(start)]
    hits = {k: [] for k in ("pick:tendency", "pick:meta only", "ban:tendency", "ban:meta only")}
    for day, day_acts in test.groupby("date"):
        tend = tendency_tables(actions, day, champs)
        for (sid, gid), g in day_acts.groupby(["series_id", "game_id"]):
            g = g.sort_values("draft_order")
            gnum = g["game_number"].iloc[0]
            locked = a.loc[(a["series_id"] == sid) & (a["game_number"] < gnum) & (a["action"] == "pick"), "ci"]
            avail = np.ones(len(champs))
            avail[locked.to_numpy()] = 0
            for r in g.itertuples():
                if r.action == "pick":
                    phase = 1 if r.draft_order in PHASE1_PICKS else 2
                    taken = set(g.loc[(g["team_id"] == r.team_id) & (g["action"] == "pick")
                                      & (g["draft_order"] < r.draft_order), "role"])
                    open_r = [x for x in ROLES if x not in taken]
                    if r.role in ROLES:
                        for name, lam in (("pick:tendency", TEAM_WEIGHT), ("pick:meta only", 0.0)):
                            p = pick_probs(tend, r.team_id, phase, open_r, avail, team_weight=lam)
                            hits[name].append(_rank(p, r.ci))
                else:
                    phase = 1 if r.draft_order <= 6 else 2
                    for name, w in (("ban:tendency", BAN_WEIGHTS), ("ban:meta only", {"team": 0, "meta": 1, "target": 0})):
                        hits[name].append(_rank(ban_probs(tend, r.team_id, r.opponent_id, phase, avail, w), r.ci))
                avail[r.ci] = 0
    rows = []
    for name, ranks in hits.items():
        ranks = np.array(ranks)
        kind, method = name.split(":")
        rows.append({"kind": kind, "method": method, "predictions": len(ranks),
                     "games": int(test["game_id"].nunique()), "top1": float((ranks == 1).mean()),
                     "top5": float((ranks <= 5).mean())})
    return pd.DataFrame(rows)


def _rank(p: np.ndarray, actual: int) -> int:
    return int((p > p[actual]).sum()) + 1


def load_likely_accuracy(live: M.Live, rebuild: bool = False) -> pd.DataFrame:
    """likely_accuracy, cached in data_cache/ (aggregate numbers only)."""
    import joblib
    key = (str(live.cutoff.date()), TEAM_WEIGHT, OTHER_PHASE_WEIGHT, tuple(BAN_WEIGHTS.values()), TENDENCY_DAYS)
    if not rebuild and LIKELY_EVAL_PATH.exists():
        saved = joblib.load(LIKELY_EVAL_PATH)
        if saved.get("key") == key:
            return saved["table"]
    table = likely_accuracy(live)
    LIKELY_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "table": table}, LIKELY_EVAL_PATH)
    return table


if __name__ == "__main__":
    import time
    t0 = time.time()
    print(load_likely_accuracy(M.live_context(), rebuild=True).to_string(index=False))
    print(f"({time.time() - t0:.0f}s)")
