"""Cloud9 Kia draft scouting vs its 3 most frequent 2026 LCS opponents.

Run:  python notebooks/01_cloud9_scouting.py
Needs data_cache/drafts.parquet (built on first run by load_drafts()).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draft_analyzer.analysis import (  # noqa: E402
    DEFAULT_BAN_WEIGHTS, bans_against, bans_by, champion_pool, priority_picks, recommend_bans, team_games,
)
from draft_analyzer.config import CLOUD9_LOL_TEAM  # noqa: E402
from draft_analyzer.drafts import load_drafts  # noqa: E402

HALF_LIFE = 3  # patches: a game 3 patches old counts half
TOP = 10
BAN_WEIGHTS = {**DEFAULT_BAN_WEIGHTS, "half_life": HALF_LIFE}  # edit here to try other weightings

pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 120)
pd.set_option("display.float_format", lambda x: f"{x:.2f}")


def show(title: str, df: pd.DataFrame, rows: int = TOP) -> None:
    print(f"\n--- {title} ---")
    print(df.head(rows).to_string(index=False) if len(df) else "  (none)")


def pct(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        df[c] = (df[c] * 100).round(0).astype(int).astype(str) + "%"
    return df


drafts = load_drafts()

# Name -> ID once; everything after joins on IDs.
c9_ids = drafts.loc[drafts["team_name"] == CLOUD9_LOL_TEAM, "team_id"].unique()
assert len(c9_ids) == 1, c9_ids
C9 = c9_ids[0]

c9_games = team_games(C9, drafts)
lcs26 = c9_games[c9_games["event"].astype(str).str.startswith("LCS") & (pd.to_datetime(c9_games["date"]).dt.year == 2026)]
opponents = lcs26.groupby("opponent_id").agg(name=("opponent_name", "last"), games=("won", "size"),
                                              c9_wins=("won", "sum")).sort_values("games", ascending=False)

print(f"{CLOUD9_LOL_TEAM} (team {C9}): {len(c9_games)} games, "
      f"{c9_games['won'].sum()}-{(~c9_games['won']).sum()}; {len(lcs26)} in 2026 LCS")
print(f"Recency: half-life {HALF_LIFE} patches; latest patch {c9_games.sort_values('patches_ago')['patch'].iloc[0]}")
print("\nMost frequent 2026 LCS opponents:")
print(opponents.head(3).to_string())

# --- Cloud9 itself -------------------------------------------------------------
print(f"\n================ {CLOUD9_LOL_TEAM} ================")
show("Champion pool (sorted by recency-weighted games)",
     pct(champion_pool(C9, half_life=HALF_LIFE, drafts=drafts), ["win_rate", "pick_rate", "weighted_pick_rate"]), 15)
pp = priority_picks(C9, half_life=HALF_LIFE, drafts=drafts)
for pos in ("first pick", "second pick"):
    show(f"Priority picks when C9 has {pos}",
         pct(pp[pp["draft_position"] == pos], ["very_first_rate", "phase1_rate", "weighted_phase1_rate"]))
show("Bans by C9", pct(bans_by(C9, half_life=HALF_LIFE, drafts=drafts), ["ban_rate", "weighted_ban_rate"]))
show("Bans against C9", pct(bans_against(C9, half_life=HALF_LIFE, drafts=drafts), ["ban_rate", "weighted_ban_rate"]))

# --- Opponents -------------------------------------------------------------------
for opp_id, opp in opponents.head(3).iterrows():
    print(f"\n================ {opp['name']} (team {opp_id}) ================")
    print(f"vs C9 in 2026 LCS: {opp['games']} games, C9 {opp['c9_wins']}-{opp['games'] - opp['c9_wins']}")
    show("Champion pool", pct(champion_pool(opp_id, half_life=HALF_LIFE, drafts=drafts),
                              ["win_rate", "pick_rate", "weighted_pick_rate"]))
    pp = priority_picks(opp_id, half_life=HALF_LIFE, drafts=drafts)
    for pos in ("first pick", "second pick"):
        show(f"Priority picks with {pos}",
             pct(pp[pp["draft_position"] == pos], ["very_first_rate", "phase1_rate", "weighted_phase1_rate"]), 5)
    show("Their bans", pct(bans_by(opp_id, half_life=HALF_LIFE, drafts=drafts), ["ban_rate", "weighted_ban_rate"]), 5)
    rec = recommend_bans(C9, opp_id, n=5, weights=BAN_WEIGHTS, drafts=drafts)
    show(f"Recommended C9 bans vs {opp['name']}",
         pct(rec, ["smoothed_wr", "weighted_pick_rate"])[["champion", "score", "smoothed_wr", "reason"]], 5)

print(f"\nBan weights: {BAN_WEIGHTS}")
