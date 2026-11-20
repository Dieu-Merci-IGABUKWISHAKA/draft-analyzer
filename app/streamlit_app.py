"""Draft scouting dashboard for coaches.

Run:  python -m streamlit run app/streamlit_app.py

All numbers come from draft_analyzer.analysis; this file only lays them out.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draft_analyzer import analysis as A  # noqa: E402
from draft_analyzer.config import CLOUD9_LOL_TEAM  # noqa: E402
from draft_analyzer.drafts import load_drafts  # noqa: E402

st.set_page_config(page_title="Draft Scouting", layout="wide")

ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]
ROLE_COLORS = {"top": "#4C78A8", "jungle": "#54A24B", "mid": "#E45756", "bot": "#F58518", "support": "#B279A2"}
PHASE_COLORS = {"Phase 1 (bans 1-6)": "#4C78A8", "Phase 2 (bans 7-10)": "#F58518"}


# --- cached data access (primitive args only; drafts come from the cached loader) ---
@st.cache_data(show_spinner="Loading drafts...")
def get_drafts() -> pd.DataFrame:
    return load_drafts()


@st.cache_data
def team_names() -> pd.Series:
    df = get_drafts()
    return df.sort_values("date").groupby("team_id")["team_name"].last().astype(str).sort_values()


@st.cache_data
def team_options(team_id):
    return A.team_options(team_id, drafts=get_drafts())


@st.cache_data
def summary():
    return A.data_summary(drafts=get_drafts())


@st.cache_data
def record(team_id, last_patches, vs=None):
    return A.record(team_id, last_patches, vs=vs, drafts=get_drafts())


@st.cache_data
def champion_pool(team_id, last_patches, half_life):
    return A.champion_pool(team_id, last_patches, half_life, drafts=get_drafts())


@st.cache_data
def priority_picks(team_id, last_patches, half_life):
    return A.priority_picks(team_id, last_patches, half_life, drafts=get_drafts())


@st.cache_data
def bans_by(team_id, last_patches, half_life, vs=None):
    return A.bans_by(team_id, last_patches, half_life, drafts=get_drafts(), vs=vs)


@st.cache_data
def recommend_bans(our_id, opp_id, n, weights, last_patches):
    return A.recommend_bans(our_id, opp_id, n=n, weights=dict(weights), drafts=get_drafts(),
                            last_patches=last_patches)


# --- helpers for display only ------------------------------------------------------
def wl(rec: dict) -> str:
    return f"{rec['wins']}-{rec['losses']}" if rec["games"] else "no games"


def pct_of(rec: dict) -> str:
    return f"{rec['wins'] / rec['games']:.0%} of {rec['games']} games" if rec["games"] else "-"


def bans_chart(bans: pd.DataFrame, top: int = 12):
    data = bans.head(top).melt(id_vars=["champion", "bans", "n_games"], value_vars=["phase1_bans", "phase2_bans"],
                               var_name="phase", value_name="count")
    data["phase"] = data["phase"].map({"phase1_bans": "Phase 1 (bans 1-6)", "phase2_bans": "Phase 2 (bans 7-10)"})
    fig = px.bar(data, x="count", y="champion", color="phase", orientation="h",
                 color_discrete_map=PHASE_COLORS, category_orders={"champion": list(bans.head(top)["champion"])},
                 hover_data={"bans": True, "n_games": True})
    fig.update_layout(height=40 + 28 * min(top, len(bans)), margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", y=1.08, title=None), xaxis_title="bans", yaxis_title=None)
    return fig


BAN_COLUMNS = {
    "champion": st.column_config.TextColumn("Champion"),
    "bans": st.column_config.NumberColumn("Bans"),
    "phase1_bans": st.column_config.NumberColumn("Phase 1"),
    "phase2_bans": st.column_config.NumberColumn("Phase 2"),
    "n_games": st.column_config.NumberColumn("Games", help="Games in the window (the sample size)"),
    "ban_rate": st.column_config.ProgressColumn("Ban rate", format="percent", min_value=0, max_value=1),
    "weighted_ban_rate": st.column_config.ProgressColumn("Recent ban rate", format="percent", min_value=0,
                                                         max_value=1, help="Weighted toward recent patches"),
    "small_sample": st.column_config.CheckboxColumn("Small sample", help=f"Fewer than {A.SMALL_SAMPLE} games"),
}


def show_bans(bans: pd.DataFrame, empty_msg: str) -> None:
    if bans.empty:
        st.info(empty_msg)
        return
    st.plotly_chart(bans_chart(bans), width="stretch")
    with st.expander("Table"):
        st.dataframe(bans, column_config=BAN_COLUMNS, hide_index=True, width="stretch")


# --- sidebar -------------------------------------------------------------------------
names = team_names()
info = summary()

with st.sidebar:
    st.header("Matchup")
    our_ids = list(names.index)
    c9 = names[names == CLOUD9_LOL_TEAM]
    our_id = st.selectbox("Our team", our_ids, index=our_ids.index(c9.index[0]) if len(c9) else 0,
                          format_func=lambda t: names[t])
    opts = team_options(our_id)
    opp_id = st.selectbox(
        "Opponent", list(opts.index),
        format_func=lambda t: f"{opts.at[t, 'name']}  ({opts.at[t, 'games_vs']} vs us)"
                              + ("" if opts.at[t, "in_league"] else "  - not 2026 LCS"),
        help="2026 LCS teams first, sorted by games played against our team.",
    )

    st.header("Recency")
    n_patches = st.slider("Last N patches", 1, info["n_patches"], info["n_patches"],
                          help="Counted back from each team's own most recent patch. Max = all data.")
    last_patches = None if n_patches == info["n_patches"] else n_patches
    half_life = st.slider("Half-life (patches)", 1.0, 10.0, float(A.DEFAULT_BAN_WEIGHTS["half_life"]), 0.5,
                          help="For 'recent' rates and the ban score: a game this many patches old counts half.")

    st.header("Ban score weights")
    d = A.DEFAULT_BAN_WEIGHTS
    if st.button("Reset to defaults"):
        for k in d:
            st.session_state.pop(f"w_{k}", None)
    weights = {
        "win_rate": st.slider("Win-rate edge", 0.0, 6.0, d["win_rate"], 0.25, key="w_win_rate",
                              help="How much their win rate on a champion matters. Below the pivot it counts against banning."),
        "pick": st.slider("Pick frequency", 0.0, 2.0, d["pick"], 0.1, key="w_pick",
                          help="Credit for how often they play it, regardless of results."),
        "banned_by_others": st.slider("Banned by other teams", 0.0, 1.0, d["banned_by_others"], 0.05,
                                      key="w_banned_by_others",
                                      help="Credit if other teams often ban it against them."),
        "our_pick_penalty": st.slider("We play it too (penalty)", 0.0, 1.0, d["our_pick_penalty"], 0.05,
                                      key="w_our_pick_penalty",
                                      help="Lower the score of champions our team also picks."),
        "wr_pivot": st.slider("Win-rate pivot", 0.40, 0.60, d["wr_pivot"], 0.01, key="w_wr_pivot",
                              help="Win rate at which the win-rate term is zero."),
        "prior_games": st.slider("Smoothing (imaginary games)", 0.0, 20.0, d["prior_games"], 1.0,
                                 key="w_prior_games",
                                 help="Average games added to every record so a 2-0 isn't treated as 100%."),
        "half_life": half_life,
    }

our_name, opp_name = names[our_id], opts.at[opp_id, "name"]

# --- header -----------------------------------------------------------------------------
st.title(f"{our_name} vs {opp_name}")
h2h, ours, theirs = record(our_id, last_patches, opp_id), record(our_id, last_patches), record(opp_id, last_patches)
window = "all data" if last_patches is None else f"each team's last {last_patches} patches"
st.caption(f"Window: {window}")
c1, c2, c3 = st.columns(3)
c1.metric("Head-to-head", wl(h2h), pct_of(h2h) if h2h["games"] else "no meetings in window", delta_color="off")
c2.metric(f"{our_name} overall", wl(ours), pct_of(ours), delta_color="off")
c3.metric(f"{opp_name} overall", wl(theirs), pct_of(theirs), delta_color="off")

tab_scout, tab_vs, tab_plan = st.tabs(["Opponent scouting", "Against us", "Ban plan"])

# --- tab: opponent scouting --------------------------------------------------------------
with tab_scout:
    pool = champion_pool(opp_id, last_patches, half_life)
    st.subheader(f"{opp_name} champion pool")
    if pool.empty:
        st.info("No games in this window.")
    else:
        role = st.radio("Role", ["all"] + ROLE_ORDER, horizontal=True, key="pool_role")
        shown = pool if role == "all" else pool[pool["role"] == role]
        top = shown.sort_values(["games", "weighted_pick_rate"], ascending=False).head(15)
        fig = px.bar(top, x="games", y="champion", color="role", orientation="h", text="record",
                     color_discrete_map=ROLE_COLORS, category_orders={"champion": list(top["champion"]),
                                                                      "role": ROLE_ORDER},
                     hover_data={"win_rate": ":.0%", "weighted_pick_rate": ":.0%", "last_played": True})
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(height=60 + 28 * len(top), margin=dict(l=0, r=40, t=10, b=0),
                          legend=dict(orientation="h", y=1.06, title=None), xaxis_title="games played (label = W-L)",
                          yaxis_title=None)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Out of {pool['n_team_games'].iloc[0]} {opp_name} games in the window. "
                   "Roles are inferred (see footer).")
        st.dataframe(
            shown, hide_index=True, width="stretch",
            column_order=["champion", "role", "games", "record", "win_rate", "pick_rate", "weighted_pick_rate",
                          "last_played", "last_patch", "small_sample"],
            column_config={
                "champion": "Champion", "role": "Role (inferred)", "games": "Games", "record": "W-L",
                "win_rate": st.column_config.ProgressColumn("Win rate", format="percent", min_value=0, max_value=1),
                "pick_rate": st.column_config.ProgressColumn("Pick rate", format="percent", min_value=0, max_value=1),
                "weighted_pick_rate": st.column_config.ProgressColumn("Recent pick rate", format="percent",
                                                                      min_value=0, max_value=1),
                "last_played": "Last played", "last_patch": "Last patch",
                "small_sample": st.column_config.CheckboxColumn("Small sample", help=f"< {A.SMALL_SAMPLE} games"),
            })

    st.subheader("Priority picks (first round of picks)")
    pp = priority_picks(opp_id, last_patches, half_life)
    cols = st.columns(2)
    for col, pos, label in [(cols[0], "first pick", "When they have first pick"),
                            (cols[1], "second pick", "When they pick second")]:
        part = pp[pp["draft_position"] == pos]
        n = int(part["n_games"].iloc[0]) if len(part) else 0
        col.markdown(f"**{label}** - {n} games")
        if part.empty:
            col.info("No games.")
            continue
        col.dataframe(
            part.head(10), hide_index=True, width="stretch",
            column_order=["champion", "very_first_picks", "phase1_picks", "phase1_rate", "record", "small_sample"],
            column_config={
                "champion": "Champion",
                "very_first_picks": st.column_config.NumberColumn("Very first pick",
                                                                  help="Times it was their first champion of the draft"),
                "phase1_picks": st.column_config.NumberColumn("Round-1 picks", help="Picked in slots 7-12"),
                "phase1_rate": st.column_config.ProgressColumn("Round-1 rate", format="percent", min_value=0, max_value=1),
                "record": "W-L",
                "small_sample": st.column_config.CheckboxColumn("Small sample", help=f"< {A.SMALL_SAMPLE} picks"),
            })

    st.subheader(f"Bans {opp_name} makes")
    st.caption("Phase 1 bans usually remove the strongest champions; phase 2 bans usually counter picks already made.")
    show_bans(bans_by(opp_id, last_patches, half_life), "No bans in this window.")

# --- tab: against us ----------------------------------------------------------------------
with tab_vs:
    st.caption(f"Only games between {our_name} and {opp_name}: {h2h['games']} in the window. "
               "Head-to-head samples are small, so read these as hints.")
    left, right = st.columns(2)
    with left:
        st.subheader(f"{opp_name} bans vs {our_name}")
        show_bans(bans_by(opp_id, last_patches, half_life, vs=our_id), "No meetings in this window.")
    with right:
        st.subheader(f"{our_name} bans vs {opp_name}")
        show_bans(bans_by(our_id, last_patches, half_life, vs=opp_id), "No meetings in this window.")

# --- tab: ban plan --------------------------------------------------------------------------
with tab_plan:
    recs = recommend_bans(our_id, opp_id, 5, tuple(sorted(weights.items())), last_patches)
    st.subheader(f"Top 5 bans against {opp_name}")
    if recs.empty:
        st.info(f"No {opp_name} games in this window.")
    else:
        n_opp = int(recs["n_opp_games"].iloc[0])
        st.caption(f"Based on {n_opp} {opp_name} games. Ranking uses the weights in the sidebar.")
        cards = st.columns(len(recs))
        for i, (card, r) in enumerate(zip(cards, recs.itertuples())):
            with card.container(border=True):
                st.markdown(f"#### {i + 1}. {r.champion}")
                st.metric("Score", f"{r.score:.2f}")
                st.markdown(f"**{r.record}** in {int(r.picks)} picks  \nSmoothed WR **{r.smoothed_wr:.0%}**")
                if r.picks < A.SMALL_SAMPLE:
                    st.markdown(":orange[Small sample]")
                st.caption(r.reason)
        with st.expander("How the score works"):
            st.markdown(
                "- **Win-rate edge**: how often they pick the champion (recent games count more) times how far "
                "their smoothed win rate is above the pivot. Below the pivot this *lowers* the score: banning a "
                "champion they lose on helps them.\n"
                "- **Pick frequency**: optional credit for how often they play it at all.\n"
                "- **Banned by other teams**: credit if other teams often ban it against them.\n"
                "- **We play it too**: banning a champion we also want costs us, so the score drops.\n"
                "- **Smoothing** adds average games to every record, so a 2-0 isn't treated as unbeatable."
            )
            st.json(weights)

# --- footer --------------------------------------------------------------------------------
st.divider()
win_range = (f"{opp_name}: patches {theirs['first_patch']}-{theirs['last_patch']}" if theirs["games"] else "")
st.caption(
    f"**Roles are inferred**: the data has no role field. Roles come from roster order in early-2025 events and "
    f"from each player's usual role and champion elsewhere (100% accurate on held-out labelled games). "
    f"**Data**: GRID official match data (Sky's The Limit 2026 hackathon), {info['games']} pro games, "
    f"patches {info['first_patch']}-{info['last_patch']} ({info['first_date']} to {info['last_date']}). "
    f"{win_range}"
)
