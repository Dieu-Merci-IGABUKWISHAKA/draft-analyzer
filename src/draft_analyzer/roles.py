"""Inferred player roles (top/jungle/mid/bot/support) for every LoL team-game.

Neither the GRID feeds nor meta/ have a role field. In some events the
roster order (feed player order = meta/rosters.csv row order) is
top, jungle, mid, bot, support; in later events it's an arbitrary but
fixed per-team order. So:

1. In LABELLED_EVENTS, roster order is the role (ground truth).
2. From those, learn P(role | champion) and each player's role history.
3. Every other team-game gets the one-player-per-role assignment that
   maximises  log P(role | player) + log P(role | champion),
   then player priors are re-estimated from those assignments and the
   assignment repeated once.

Result: one row per (game_id, team_id, player_id) with role_inferred and
role_source ("roster_order" or "inferred").
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import permutations

import numpy as np
import pandas as pd

from draft_analyzer.config import LOL_DIR, META_DIR

ROLES = ["top", "jungle", "mid", "bot", "support"]

# Events whose roster order matches top/jungle/mid/bot/support. Checked by
# where signature champions sit: junglers (Vi, Lee Sin, ...) in slot 2,
# supports (Rell, Nautilus, ...) in slot 5, ADCs in slot 4 in >=90% of games.
LABELLED_EVENTS = {
    "LCK - LCK Cup 2025",
    "LEC - Winter 2025",
    "LEC - Spring 2025",
    "LTA Cross-Conference - Split 1 2025",
    "LTA North - Split 1 2025",
    "LTA South - Split 1 2025",
}

_CHAMP_ALPHA = 1.0  # pseudo-counts spread evenly over roles for P(role | champion)
_PLAYER_ALPHA = 0.5  # same for P(role | player)


def load_rosters() -> pd.DataFrame:
    r = pd.read_csv(META_DIR / "rosters.csv", dtype=str, keep_default_na=False)
    r = r[r["title"] == "lol"].copy()
    r["slot"] = r.groupby(["game_id", "team_id"]).cumcount()
    # Some rosters list a 6th (substitute) player with no champion, sometimes
    # mid-list, so roster order isn't trusted there (team_size != 5). Drop them.
    r["team_size"] = r.groupby(["game_id", "team_id"])["player_id"].transform("size")
    r = r[r["character_id"] != ""]
    series = pd.read_csv(LOL_DIR / "series.csv", dtype=str, keep_default_na=False)
    return r.merge(series[["series_id", "event"]], on="series_id", how="left")


def _prob_table(counts: dict[str, Counter], alpha: float) -> dict[str, np.ndarray]:
    out = {}
    for key, c in counts.items():
        v = np.array([c.get(role, 0.0) for role in ROLES]) + alpha / len(ROLES)
        out[key] = v / v.sum()
    return out


def _assign(group: pd.DataFrame, player_p: dict, champ_p: dict) -> list[str]:
    uniform = np.full(len(ROLES), 1 / len(ROLES))
    scores = np.array(
        [
            np.log(player_p.get(p, uniform)) + np.log(champ_p.get(c, uniform))
            for p, c in zip(group["player_id"], group["character_id"], strict=False)
        ]
    )
    # A rare glitched roster lists 6 players with champions. Extra "no role" columns
    # let the leftover player be assigned somewhere; they score very low, so they
    # are used only when there is no other choice.
    n_extra = max(0, len(group) - len(ROLES))
    scores = np.hstack([scores, np.full((len(group), n_extra), np.log(1e-4))])
    best, best_score = None, -np.inf
    for perm in permutations(range(scores.shape[1]), len(group)):
        s = scores[np.arange(len(group)), perm].sum()
        if s > best_score:
            best, best_score = perm, s
    return [ROLES[i] if i < len(ROLES) else None for i in best]


def infer_roles(rosters: pd.DataFrame | None = None, labelled_events: set[str] = LABELLED_EVENTS) -> pd.DataFrame:
    r = load_rosters() if rosters is None else rosters.copy()
    labelled = r["event"].isin(labelled_events) & (r["team_size"] == 5)
    r.loc[labelled, "role_inferred"] = r.loc[labelled, "slot"].map(dict(enumerate(ROLES)))

    champ_counts: dict[str, Counter] = defaultdict(Counter)
    for c, role in zip(r.loc[labelled, "character_id"], r.loc[labelled, "role_inferred"], strict=False):
        champ_counts[c][role] += 1
    champ_p = _prob_table(champ_counts, _CHAMP_ALPHA)

    # Initial player prior: hard labels where known, soft champion votes elsewhere.
    player_counts: dict[str, Counter] = defaultdict(Counter)
    for p, role in zip(r.loc[labelled, "player_id"], r.loc[labelled, "role_inferred"], strict=False):
        player_counts[p][role] += 1
    for p, c in zip(r.loc[~labelled, "player_id"], r.loc[~labelled, "character_id"], strict=False):
        for role, prob in zip(ROLES, champ_p.get(c, np.full(5, 0.2)), strict=False):
            player_counts[p][role] += prob

    todo = r[~labelled]
    for _ in range(2):
        player_p = _prob_table(player_counts, _PLAYER_ALPHA)
        assigned = {}
        for _, g in todo.groupby(["game_id", "team_id"], sort=False):
            assigned.update(zip(g.index, _assign(g, player_p, champ_p), strict=False))
        r.loc[list(assigned), "role_inferred"] = pd.Series(assigned)
        player_counts = defaultdict(Counter)
        for p, role in zip(r["player_id"], r["role_inferred"], strict=False):
            player_counts[p][role] += 1

    r["role_source"] = np.where(labelled, "roster_order", "inferred")
    return r[["game_id", "team_id", "player_id", "player_name", "character_id", "role_inferred", "role_source"]]
