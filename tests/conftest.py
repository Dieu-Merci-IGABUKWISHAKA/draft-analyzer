"""Shared fixtures. Tests use a small SYNTHETIC drafts table (made-up teams and
champions), so they run without the licensed GRID data and never contain it."""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from draft_analyzer.drafts import COLUMNS, DRAFT_TEMPLATE  # noqa: E402

ROLES = ["top", "jungle", "mid", "bot", "support"]


def make_synthetic_drafts(n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Six teams, 40 champions (8 per role), two games a day, a new patch every 10 days.
    Stronger teams (lower index) win more; a few champions are slightly overpowered."""
    rng = np.random.default_rng(seed)
    teams = [(f"T{i}", f"Team {i}") for i in range(6)]
    champs = {role: [f"C{r * 8 + j}" for j in range(8)] for r, role in enumerate(ROLES)}
    strong = {"C0", "C9", "C18"}
    rows = []
    game_no = 0
    for day in range(n_days):
        d = date(2025, 1, 1) + timedelta(days=day)
        patch = f"15.{day // 10 + 1}"
        for _ in range(2):
            game_no += 1
            i, j = rng.choice(len(teams), size=2, replace=False)
            (fp_id, fp_name), (sp_id, sp_name) = teams[i], teams[j]
            fp_side = "blue" if rng.random() < 0.7 else "red"
            sides = {fp_id: fp_side, sp_id: "red" if fp_side == "blue" else "blue"}
            names = {fp_id: fp_name, sp_id: sp_name}
            opp = {fp_id: sp_id, sp_id: fp_id}
            pool = [c for cs in champs.values() for c in cs]
            used: set[str] = set()
            picks_by_team: dict[str, list[tuple[str, str]]] = {fp_id: [], sp_id: []}
            actions = []
            for slot, (who, kind) in enumerate(DRAFT_TEMPLATE, 1):
                team = fp_id if who == "F" else sp_id
                if kind == "pick":
                    role = ROLES[len(picks_by_team[team])]
                    options = [c for c in champs[role] if c not in used]
                    champ = str(rng.choice(options))
                    picks_by_team[team].append((champ, role))
                else:
                    role = None
                    champ = str(rng.choice([c for c in pool if c not in used]))
                used.add(champ)
                actions.append((slot, team, kind, champ, role))
            strength = {fp_id: -0.3 * i, sp_id: -0.3 * j}
            for t in (fp_id, sp_id):
                strength[t] += 0.4 * sum(c in strong for c, _ in picks_by_team[t])
            p_fp = 1 / (1 + np.exp(-(strength[fp_id] - strength[sp_id])))
            winner = fp_id if rng.random() < p_fp else sp_id
            for slot, team, kind, champ, role in actions:
                rows.append(
                    {
                        "series_id": f"S{game_no}",
                        "game_id": f"G{game_no}",
                        "game_number": 1,
                        "event": "Synthetic League",
                        "stage": "Regular",
                        "date": d,
                        "patch": patch,
                        "team_id": team,
                        "team_name": names[team],
                        "opponent_id": opp[team],
                        "opponent_name": names[opp[team]],
                        "side": sides[team],
                        "action": kind,
                        "draft_order": slot,
                        "champion_id": champ,
                        "champion_name": f"Champ{champ[1:]}",
                        "game_winner_team_id": winner,
                        "player_id": f"P-{team}-{role}" if role else None,
                        "player_name": f"{team}-{role}" if role else None,
                        "role_inferred": role,
                        "role_source": "inferred" if role else None,
                    }
                )
    df = pd.DataFrame(rows, columns=COLUMNS)
    for col in (
        "event",
        "stage",
        "patch",
        "side",
        "action",
        "team_name",
        "opponent_name",
        "champion_name",
        "role_inferred",
        "role_source",
    ):
        df[col] = df[col].astype("category")
    return df


@pytest.fixture(scope="session")
def synthetic_drafts() -> pd.DataFrame:
    return make_synthetic_drafts()
