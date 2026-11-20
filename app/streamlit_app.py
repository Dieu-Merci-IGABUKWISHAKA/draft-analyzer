"""Draft scouting dashboard for coaches.

Run:  python -m streamlit run app/streamlit_app.py

All numbers come from draft_analyzer.analysis; this file only lays them out.
Text only: no game artwork or champion images.
"""

import html
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from draft_analyzer import analysis as A  # noqa: E402
from draft_analyzer import insights as I  # noqa: E402
from draft_analyzer import macro as MA  # noqa: E402
from draft_analyzer import backtest as BT  # noqa: E402
from draft_analyzer import model as M  # noqa: E402
from draft_analyzer import prep as P  # noqa: E402
from draft_analyzer import simulator as S  # noqa: E402
from draft_analyzer.config import CLOUD9_LOL_TEAM  # noqa: E402
from draft_analyzer.drafts import load_drafts  # noqa: E402

st.set_page_config(page_title="Draft Scouting", page_icon="🎯", layout="wide")

DEFAULT_OPPONENT = "LYON"
DEFAULT_LAST_PATCHES = 6
MIN_POOL_GAMES = 3  # champion pool hides fewer games than this unless "Show small samples"
ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]
ROLE_COLORS = {"top": "#4C78A8", "jungle": "#54A24B", "mid": "#E45756", "bot": "#F58518", "support": "#B279A2"}
PHASE_COLORS = {"Phase 1 (bans 1-6)": "#4C78A8", "Phase 2 (bans 7-10)": "#F58518"}
GREY = "#9aa0a6"

st.markdown("""
<style>
.ban-name {font-size: 1.2rem; font-weight: 600; white-space: nowrap; overflow: hidden;
           text-overflow: ellipsis; margin-bottom: 0.25rem;}
.ban-line {margin: 0.1rem 0; font-size: 0.85rem; line-height: 1.35;}
.ban-small {color: #d9822b;}
.chip {display: inline-block; padding: 0.1rem 0.5rem; margin: 0.1rem; border: 1px solid #9aa0a6;
       border-radius: 0.4rem; font-size: 0.85rem; min-width: 4.5rem; text-align: center;}
.chip-next {border: 2px solid #f0b400; font-weight: 600;}
.chip-empty {color: #9aa0a6;}
.opt-ev {font-size: 0.8rem; color: #8a8f98; line-height: 1.3;}
.badge {display: inline-block; padding: 0 0.45rem; border-radius: 0.6rem; background: #eef2f6; color: #4a5562;
        font-size: 0.75rem; white-space: nowrap;}
.ban-lead {font-weight: 600;}
</style>
""", unsafe_allow_html=True)


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
def window_patches(team_ids, last_patches):
    return A.window_patches(list(team_ids), last_patches, drafts=get_drafts())


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
def fearless_events():
    return A.fearless_events(drafts=get_drafts())


@st.cache_data
def fearless_by_game(team_id, last_patches):
    return A.fearless_by_game(team_id, last_patches, drafts=get_drafts())


@st.cache_data
def champion_options():
    return A.champion_options(drafts=get_drafts())


@st.cache_data
def series_plan(our_id, opp_id, used_by_us, used_by_them, rule, weights, last_patches, opp_name="them"):
    return A.series_plan(our_id, opp_id, set(used_by_us), set(used_by_them), rule=rule, n=5,
                         weights=dict(weights), last_patches=last_patches, drafts=get_drafts(),
                         opponent_name=opp_name)


@st.cache_data
def recommend_bans(our_id, opp_id, n, weights, last_patches, opp_name="them"):
    return A.recommend_bans(our_id, opp_id, n=n, weights=dict(weights), drafts=get_drafts(),
                            last_patches=last_patches, opponent_name=opp_name)


@st.cache_resource(show_spinner="Preparing insights...")
def get_scout():
    return I.Scout(load_drafts())


@st.cache_resource(show_spinner="Reading early-game events...")
def get_macro():
    return MA.Macro(get_scout())


@st.cache_data(show_spinner="Checking which early-game patterns hold up in later games...")
def macro_replication() -> pd.DataFrame:
    return MA.load_replication(get_macro())


@st.cache_data(show_spinner="Summarising the early game...")
def macro_summary(team):
    return get_macro().summary(team, I.DEFAULT_PATCHES, replication=macro_replication())


# --- display helpers -----------------------------------------------------------------
def wl(rec: dict) -> str:
    return f"{rec['wins']}–{rec['losses']}" if rec["games"] else "no games"


def pct_of(rec: dict) -> str:
    return f"{rec['wins'] / rec['games']:.0%} of {rec['games']} games" if rec["games"] else "–"


def what_this_means(text: str) -> None:
    st.markdown(f"**What this means:** {text}")


def text_bar(rate: float, width: int = 10) -> str:
    filled = round(rate * width)
    return "▰" * filled + "▱" * (width - filled) + f"  {rate:.0%}"


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
    "ban_rate": st.column_config.ProgressColumn("Ban rate", format="%.0f%%", min_value=0, max_value=100),
    "weighted_ban_rate": st.column_config.ProgressColumn("Recent ban rate", format="%.0f%%", min_value=0,
                                                         max_value=100, help="Weighted toward recent patches"),
    "small_sample": st.column_config.CheckboxColumn("Small sample", help=f"Fewer than {A.SMALL_SAMPLE} games"),
}


def show_bans(bans: pd.DataFrame, empty_msg: str) -> None:
    if bans.empty:
        st.info(empty_msg)
        return
    st.plotly_chart(bans_chart(bans), width="stretch")
    with st.expander("Table"):
        st.dataframe(bans.assign(ban_rate=bans["ban_rate"] * 100, weighted_ban_rate=bans["weighted_ban_rate"] * 100),
                     column_config=BAN_COLUMNS, hide_index=True, width="stretch")


def ban_card(rank: int, r) -> None:
    with st.container(border=True):
        name = html.escape(str(r.champion))
        st.markdown(f'<div class="ban-name" title="{name}">{rank}. {name}</div>', unsafe_allow_html=True)
        for i, line in enumerate(r.reason_lines):      # biggest share of the score first
            text = html.escape(line).replace(" · small sample", ' · <span class="ban-small">small sample</span>')
            st.markdown(f'<p class="ban-line{" ban-lead" if i == 0 else ""}">{text}</p>', unsafe_allow_html=True)
        st.caption(f"Ban score {r.score:.2f} · smoothed win rate {r.smoothed_wr:.0%}",
                   help="Higher = better ban. Reasons are ordered by how much they add to the score. Smoothed win "
                        "rate adds average games so small records aren't trusted fully.")


# --- sidebar -------------------------------------------------------------------------
names = team_names()
info = summary()

with st.sidebar:
    st.markdown("**Draft Tendency Analyzer**  \nScout an opponent's draft habits, plan bans and play out a draft. "
                "Every number shows its sample size, and each method was tested on later games.")
    st.header("Matchup")
    our_ids = list(names.index)
    c9 = names[names == CLOUD9_LOL_TEAM]
    our_id = st.selectbox("Our team", our_ids, index=our_ids.index(c9.index[0]) if len(c9) else 0,
                          format_func=lambda t: names[t])
    opts = team_options(our_id)
    opp_ids = list(opts.index)
    default_opp = opts.index[opts["name"] == DEFAULT_OPPONENT]
    opp_id = st.selectbox(
        "Opponent", opp_ids, index=opp_ids.index(default_opp[0]) if len(default_opp) else 0,
        format_func=lambda t: f"{opts.at[t, 'name']}  ({opts.at[t, 'games_vs']} vs us)"
                              + ("" if opts.at[t, "in_league"] else "  - not 2026 LCS"),
        help="2026 LCS teams first, sorted by games played against our team.",
    )

    st.header("Recency")
    n_patches = st.slider("Last N patches", 1, info["n_patches"], min(DEFAULT_LAST_PATCHES, info["n_patches"]),
                          help="Each team's last N patches with games. Max = all data.")
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
        "pick": st.slider("Pick frequency", 0.0, 4.0, d["pick"], 0.1, key="w_pick",
                          help="Credit for how often they play it, regardless of results."),
        "banned_by_others": st.slider("Banned by other teams", 0.0, 2.0, d["banned_by_others"], 0.05,
                                      key="w_banned_by_others",
                                      help="Credit if other teams often ban it against them."),
        "our_pick_penalty": st.slider("We play it too (penalty)", 0.0, 1.0, d["our_pick_penalty"], 0.05,
                                      key="w_our_pick_penalty",
                                      help="Lower the score of champions our team also picks."),
        "wr_pivot": st.slider("Win-rate pivot", 0.40, 0.60, d["wr_pivot"], 0.01, key="w_wr_pivot",
                              help="Win rate at which the win-rate term is zero."),
        "prior_games": st.slider("Smoothing (imaginary games)", 0.0, 30.0, d["prior_games"], 1.0,
                                 key="w_prior_games",
                                 help="Average games added to every record so a 2-0 isn't treated as 100%."),
        "min_picks": st.slider("Minimum picks", 1, 10, int(d["min_picks"]), 1, key="w_min_picks",
                               help="Ignore champions they picked fewer times than this in the window."),
        "half_life": half_life,
    }

our_name, opp_name = names[our_id], opts.at[opp_id, "name"]

# --- header -----------------------------------------------------------------------------
st.title(f"{our_name} vs {opp_name}")
h2h, ours, theirs = record(our_id, last_patches, opp_id), record(our_id, last_patches), record(opp_id, last_patches)
span = window_patches((our_id, opp_id), last_patches)
span_text = f"Patches {span[0]}–{span[1]}" if span else "No games"
window = "all data" if last_patches is None else f"each team's last {last_patches} patches"
st.markdown(f"#### {span_text}")
st.caption(f"Window: {window}")
c1, c2, c3 = st.columns(3)
c1.metric("Head-to-head", wl(h2h), pct_of(h2h) if h2h["games"] else "no meetings in window", delta_color="off", delta_arrow="off")
c2.metric(f"{our_name} overall", wl(ours), pct_of(ours), delta_color="off", delta_arrow="off")
c3.metric(f"{opp_name} overall", wl(theirs), pct_of(theirs), delta_color="off", delta_arrow="off")

tab_prep, tab_sim, tab_scout, tab_insights, tab_plan, tab_fearless, tab_vs = st.tabs(
    ["Match prep", "Draft simulator", "Opponent scouting", "Insights", "Ban plan", "Fearless series", "Against us"])

# --- tab: opponent scouting --------------------------------------------------------------
with tab_scout:
    what_this_means(f"these are the champions {opp_name} relies on and grabs early, so expect to face them "
                    f"and prepare answers for the ones they win with.")
    pool = champion_pool(opp_id, last_patches, half_life)
    st.subheader(f"{opp_name} champion pool")
    if pool.empty:
        st.info("No games in this window.")
    else:
        f1, f2 = st.columns([2, 1])
        role = f1.radio("Role", ["all"] + ROLE_ORDER, horizontal=True, key="pool_role")
        show_small = f2.checkbox("Show small samples", value=False,
                                 help=f"Include champions with fewer than {MIN_POOL_GAMES} games")
        shown = pool if role == "all" else pool[pool["role"] == role]
        n_hidden = int((shown["games"] < MIN_POOL_GAMES).sum())
        if not show_small:
            shown = shown[shown["games"] >= MIN_POOL_GAMES]
        shown = shown.sort_values(["games", "weighted_pick_rate"], ascending=False)

        if shown.empty:
            st.info("No champions with enough games here. Tick 'Show small samples' to see them.")
        else:
            top = shown.head(15)
            fig = px.bar(top, x="games", y="champion", color="role", orientation="h", text="record",
                         color_discrete_map=ROLE_COLORS,
                         category_orders={"champion": list(top["champion"]), "role": ROLE_ORDER},
                         hover_data={"win_rate": ":.0%", "weighted_pick_rate": ":.0%", "last_played": True})
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(height=60 + 28 * len(top), margin=dict(l=0, r=40, t=10, b=0),
                              legend=dict(orientation="h", y=1.06, title=None),
                              xaxis_title="games played (label = W-L)", yaxis_title=None)
            st.plotly_chart(fig, width="stretch")

            table = shown.assign(win_rate_bar=shown["win_rate"].map(text_bar),
                                 pick_rate=shown["pick_rate"] * 100,
                                 weighted_pick_rate=shown["weighted_pick_rate"] * 100)
            styled = table.style.apply(
                lambda row: [f"color: {GREY}" if c == "win_rate_bar" and row["games"] < A.SMALL_SAMPLE else ""
                             for c in row.index], axis=1)
            st.dataframe(
                styled, hide_index=True, width="stretch",
                column_order=["champion", "role", "games", "record", "win_rate_bar", "pick_rate",
                              "weighted_pick_rate", "last_played", "last_patch", "small_sample"],
                column_config={
                    "champion": "Champion", "role": "Role (inferred)", "games": "Games", "record": "W-L",
                    "win_rate_bar": st.column_config.TextColumn(
                        "Win rate", help=f"Greyed out when based on fewer than {A.SMALL_SAMPLE} games"),
                    "pick_rate": st.column_config.ProgressColumn("Pick rate", format="%.0f%%", min_value=0,
                                                                 max_value=100),
                    "weighted_pick_rate": st.column_config.ProgressColumn("Recent pick rate", format="%.0f%%",
                                                                          min_value=0, max_value=100),
                    "last_played": "Last played", "last_patch": "Last patch",
                    "small_sample": st.column_config.CheckboxColumn("Small sample",
                                                                    help=f"< {A.SMALL_SAMPLE} games"),
                })
        st.caption(f"Out of {pool['n_team_games'].iloc[0]} {opp_name} games in the window."
                   + (f" {n_hidden} champions with fewer than {MIN_POOL_GAMES} games hidden." if n_hidden and not show_small else "")
                   + " Roles are inferred (see footer).")

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
            part.head(10).assign(phase1_rate=part["phase1_rate"].head(10) * 100), hide_index=True, width="stretch",
            column_order=["champion", "very_first_picks", "phase1_picks", "phase1_rate", "record", "small_sample"],
            column_config={
                "champion": "Champion",
                "very_first_picks": st.column_config.NumberColumn("Very first pick",
                                                                  help="Times it was their first champion of the draft"),
                "phase1_picks": st.column_config.NumberColumn("Round-1 picks", help="Picked in slots 7-12"),
                "phase1_rate": st.column_config.ProgressColumn("Round-1 rate", format="%.0f%%", min_value=0, max_value=100),
                "record": "W-L",
                "small_sample": st.column_config.CheckboxColumn("Small sample", help=f"< {A.SMALL_SAMPLE} picks"),
            })

    st.subheader(f"Bans {opp_name} makes")
    st.caption("Phase 1 bans usually remove the strongest champions; phase 2 bans usually counter picks already made.")
    show_bans(bans_by(opp_id, last_patches, half_life), "No bans in this window.")

    st.subheader(f"Early game: {opp_name}")
    mac = macro_summary(opp_id)
    prof = mac["profile"]
    st.caption(f"Supporting context from the game event feeds: {mac['games']} {opp_name} games in their last "
               f"{I.DEFAULT_PATCHES} patches. Only patterns of a kind that held up in later games, and that are "
               f"confident for {opp_name}, are listed.")
    lines = []
    for r in prof[prof["show_rate"]].itertuples():
        lines.append(f"**{r.objective}**: {opp_name} take it {r.rate:.0%} of the time ({r.took} of {r.games} games; "
                     f"90% range {r.lo:.0%}–{r.hi:.0%}).")
    for r in prof[prof["show_timing"]].itertuples():
        lines.append(f"**{r.objective}**: when they take it, median {MA.mmss(r.their_median_s)} vs "
                     f"{MA.mmss(r.league_median_s)} league-wide ({r.taken_n} games).")
    lines += mac["jungler_lines"]
    if lines:
        for line in lines:
            st.markdown(f"- {line}")
    else:
        st.caption(f"No early-game pattern for {opp_name} passes the bar.")
    with st.expander("All early-game numbers, and which kinds held up"):
        st.dataframe(prof.assign(rate=prof["rate"] * 100, their=prof["their_median_s"].map(MA.mmss),
                                 league=prof["league_median_s"].map(MA.mmss),
                                 sample=prof["took"].astype(str) + " of " + prof["games"].astype(str)),
                     hide_index=True, width="stretch",
                     column_order=["objective", "rate", "sample", "their", "league"],
                     column_config={"objective": "First…", "rate": st.column_config.NumberColumn(
                                        f"{opp_name} took it", format="%.0f%%"),
                                    "sample": "Games", "their": "Their median time",
                                    "league": "League median"})
        rep = macro_replication()
        st.dataframe(rep.assign(held=[I.holds_up(a, b) for a, b in zip(rep["survives"], rep["late_lo"])],
                                same_direction=rep["same_direction"] * 100, survives=rep["survives"] * 100),
                     hide_index=True, width="stretch",
                     column_order=["kind", "objective", "checks", "same_direction", "survives", "held"],
                     column_config={"kind": "Pattern", "objective": "Objective",
                                    "checks": st.column_config.NumberColumn("Checks", help="Confident patterns"),
                                    "same_direction": st.column_config.NumberColumn("Same direction", format="%.0f%%"),
                                    "survives": st.column_config.NumberColumn("Kept", format="%.0f%%"),
                                    "held": st.column_config.CheckboxColumn("Held up")})
        st.caption("Each team's patterns from games before July 2025 and April 2026 were measured again on later "
                   "games. 'Held up' = at least 25% of the gap kept and the later gap's 90% range above zero. "
                   "Most early-game habits did not hold up; first-tower rate (mostly team strength) and the "
                   "jungler–first-grub link did. Timings are game-clock times from the feeds.")

# --- tab: against us ----------------------------------------------------------------------
with tab_vs:
    what_this_means(f"this shows what {opp_name} fears from {our_name} and what {our_name} has chosen to take "
                    f"away from them when the two teams met.")
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
    what_this_means(f"with our five bans, these take away the champions {opp_name} plays most and wins with, "
                    f"while sparing the ones we want for ourselves.")
    recs = recommend_bans(our_id, opp_id, 5, tuple(sorted(weights.items())), last_patches, opp_name)
    st.subheader(f"Top 5 bans against {opp_name}")
    st.caption(BT.headline_caption() + " Details: docs/VALIDATION.md.")
    if recs.empty:
        st.info(f"No champion qualifies: {opp_name} has no games in this window, or none picked "
                f"'Minimum picks' times. Lower the slider or widen the patch window.")
    else:
        n_opp = int(recs["n_opp_games"].iloc[0])
        st.caption(f"Based on {n_opp} {opp_name} games. Ranking uses the weights in the sidebar (defaults tuned on "
                   f"2025 games). On each card the reason that adds most to the score comes first.")
        rows = [recs.iloc[:3], recs.iloc[3:]]
        rank = 1
        for chunk in rows:
            cols = st.columns(3)
            for col, r in zip(cols, chunk.itertuples()):
                with col:
                    ban_card(rank, r)
                rank += 1
        with st.expander("How the score works"):
            st.markdown(
                "- **Win-rate edge**: how often they pick the champion (recent games count more) times how far "
                "their smoothed win rate is above the pivot. Below the pivot this *lowers* the score: banning a "
                "champion they lose on helps them.\n"
                "- **Pick frequency**: credit for how often they play it at all (the biggest part by default).\n"
                "- **Banned by other teams**: credit if other teams often ban it against them.\n"
                "- **We play it too**: banning a champion we also want costs us, so the score drops.\n"
                "- **Smoothing** adds average games to every record, so a 2-0 isn't treated as unbeatable."
            )
            st.json(weights)

# --- tab: fearless series ---------------------------------------------------------------------
with tab_fearless:
    what_this_means(f"in Fearless Draft every champion used in a series is gone for the rest of it, so by game 3 "
                    f"{opp_name} is pushed off its comfort picks; this shows how they adapt, and what the next "
                    f"game looks like once you mark what's already been played.")
    ev = fearless_events()
    fearless_ev = ev[ev["fearless"]]
    rules = sorted(fearless_ev["rule"].unique())
    rule = "hard" if "hard" in rules else (rules[0] if rules else "hard")
    with st.expander(f"Which events use Fearless Draft? {len(fearless_ev)} of {len(ev)} (detected from the data)"):
        st.caption("A champion picked again later in the same series means no Fearless. "
                   "'hard' = no champion is ever repeated by either team; 'soft' = only by the same team.")
        st.dataframe(ev, hide_index=True, width="stretch", column_config={
            "event": "Event", "multi_game_series": "Multi-game series", "games": "Games",
            "repeats_any_team": "Repeats (any team)", "repeats_same_team": "Repeats (same team)",
            "rule": "Rule", "fearless": "Fearless"})

    st.subheader(f"{opp_name} by game number (Fearless series only)")
    fb = fearless_by_game(opp_id, last_patches)
    summ = fb["summary"]
    if summ["games"].sum() == 0:
        st.info("No Fearless series games in this window.")
    else:
        cols = st.columns(len(A.FEARLESS_ORDER))
        for col, r in zip(cols, summ.itertuples()):
            col.metric(f"Game {r.game[1:]}", r.record.replace("-", "–") if r.games else "–",
                       f"{r.games} games" + (f" · {r.win_rate:.0%}" if r.games else ""), delta_color="off",
                       delta_arrow="off", help=f"{r.games} games, {r.distinct_champions} different champions")
            if r.small_sample:
                col.markdown(":orange[Small sample]", help=f"Fewer than {A.SMALL_SAMPLE} games")
        st.caption("Games 4 and 5 only happen in best-of-5 series, so their samples are small: read them as anecdotes. "
                   "Champions = how many different champions they played in that game number; it rises in later "
                   "games because Fearless removes the ones already used.")
        tp = fb["top_picks"]
        for tab, g in zip(st.tabs([f"Most picked in game {g[1:]}" for g in A.FEARLESS_ORDER]), A.FEARLESS_ORDER):
            part = tp[tp["game"] == g]
            n_games = int(summ.loc[summ["game"] == g, "games"].iloc[0])
            if part.empty:
                tab.info("No games.")
                continue
            if n_games < A.SMALL_SAMPLE:
                tab.warning(f"Only {n_games} game{'s' if n_games != 1 else ''}: too few to call this a tendency.")
            tab.dataframe(part.assign(pick_rate=part["pick_rate"] * 100), hide_index=True, width="stretch",
                          column_order=["champion", "role", "picks", "pick_rate"],
                          column_config={"champion": "Champion", "role": "Role (inferred)",
                                         "picks": st.column_config.NumberColumn("Picks", help=f"Out of {n_games} games"),
                                         "pick_rate": st.column_config.ProgressColumn(
                                             f"Of {n_games} games", format="%.0f%%", min_value=0, max_value=100)})

    st.subheader("Series planner")
    st.caption(f"Mark the champions already played earlier in this series. Rule applied: {rule} Fearless "
               f"({'gone for both teams' if rule == 'hard' else 'gone only for the team that played it'}).")
    champs = champion_options()
    p1, p2 = st.columns(2)
    used_us = p1.multiselect(f"Used by {our_name}", list(champs.index), format_func=lambda c: champs[c],
                             key=f"used_us_{our_id}_{opp_id}")
    used_them = p2.multiselect(f"Used by {opp_name}", list(champs.index), format_func=lambda c: champs[c],
                               key=f"used_them_{our_id}_{opp_id}")
    plan = series_plan(our_id, opp_id, tuple(sorted(used_us)), tuple(sorted(used_them)), rule,
                       tuple(sorted(weights.items())), last_patches, opp_name)

    lost, likely = plan["lost"], plan["likely_picks"]
    l1, l2 = st.columns(2)
    with l1:
        st.markdown(f"**What {opp_name} loses**")
        if lost.empty:
            st.caption("Nothing yet: mark champions above.")
        else:
            st.metric("Share of their recent picks now unavailable", f"{plan['lost_share']:.0%}")
            st.dataframe(lost, hide_index=True, width="stretch", column_order=["champion", "roles", "games"],
                         column_config={"champion": "Champion", "roles": "Role(s)",
                                        "games": st.column_config.NumberColumn("Games played", help="In the window")})
    with l2:
        st.markdown(f"**{opp_name}'s likely picks now**")
        if likely.empty:
            st.caption("No games in this window.")
        else:
            st.dataframe(likely.assign(weighted_pick_rate=likely["weighted_pick_rate"] * 100), hide_index=True,
                         width="stretch", column_order=["champion", "roles", "games", "weighted_pick_rate"],
                         column_config={"champion": "Champion", "roles": "Role(s)", "games": "Games played",
                                        "weighted_pick_rate": st.column_config.ProgressColumn(
                                            "Recent pick rate", format="%.0f%%", min_value=0, max_value=100)})

    st.markdown("**Ban plan for the next game**")
    bans = plan["bans"]
    if bans.empty:
        st.info("No champions qualify. Lower 'Minimum picks' or widen the patch window.")
    else:
        rank = 1
        for chunk in [bans.iloc[:3], bans.iloc[3:]]:
            cols = st.columns(3)
            for col, r in zip(cols, chunk.itertuples()):
                with col:
                    ban_card(rank, r)
                rank += 1

# --- tab: draft simulator -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the win model (first run trains it: about a minute)...")
def get_live():
    return M.live_context(drafts=load_drafts())


@st.cache_resource(show_spinner="Preparing the simulator...")
def get_sim(our, opp):
    return S.Simulator(our, opp, get_live())


@st.cache_data(show_spinner="Checking 'likely picks' against past games (first run: about a minute)...")
def likely_accuracy() -> pd.DataFrame:
    return S.load_likely_accuracy(get_live())


def sim_apply(skey, sim, setup, champ, role):
    try:
        st.session_state[skey] = sim.apply(setup, st.session_state.get(skey, ()), champ, role)
    except ValueError as e:
        st.session_state[skey + "_err"] = str(e)


def sim_set_role(skey, sim, setup, slot, widget):
    st.session_state[skey] = sim.set_role(setup, st.session_state.get(skey, ()), slot, st.session_state[widget])


def sim_undo(skey):
    st.session_state[skey] = st.session_state.get(skey, ())[:-1]


def sim_reset(skey):
    st.session_state[skey] = ()


ROLE_SHORT = {"top": "Top", "jungle": "Jgl", "mid": "Mid", "bot": "Bot", "support": "Sup"}


def chip(text: str, cls: str = "") -> str:
    return f'<span class="chip {cls}">{html.escape(text)}</span>'


with tab_sim:
    what_this_means("play a draft pick by pick and see what each choice shifts. The model's draft effect is small "
                    "(about 3 points over a whole draft) and team strength matters more, so read every number as a "
                    "nudge backed by past games, not as a prediction of who wins.")
    sim = get_sim(our_id, opp_id)
    champ_names = sim.live.names
    key = f"{our_id}_{opp_id}"
    skey = f"sim_actions_{key}"
    team_label = {"us": our_name, "them": opp_name}

    s1, s2, s3 = st.columns(3)
    our_side = s1.radio("Our side", ["blue", "red"], horizontal=True, key=f"sim_side_{key}",
                        format_func=str.capitalize)
    first = s2.radio("First pick (first selection)", ["us", "them"], horizontal=True, key=f"sim_first_{key}",
                     format_func=team_label.get,
                     help="Since 2026 the team with first selection chooses side or first pick, so set both.")
    game_no = s3.selectbox("Game in the series", [1, 2, 3, 4, 5], key=f"sim_game_{key}")
    locked = st.multiselect("Champions used in earlier games of this series (hard Fearless: locked for both teams)",
                            list(champ_names.index), format_func=champ_names.get, key=f"sim_locked_{key}",
                            disabled=game_no == 1, placeholder="Type to search...")
    if game_no == 1:
        locked = []
    with st.expander("Lineups: each team's last game. Change a player if a substitute plays."):
        lineups = {}
        for col, who in zip(st.columns(2), ("us", "them")):
            roster, default = sim.roster(who), sim.default_lineup(who)
            col.markdown(f"**{team_label[who]}**")
            label = {r.player_id: f"{r.name} ({r.role}, {r.games} games)" for r in roster.itertuples()}
            chosen = []
            for role in S.ROLES:
                opts = list(dict.fromkeys(([default[role]] if role in default else [])
                                          + list(roster.loc[roster["role"] == role, "player_id"])
                                          + list(roster["player_id"])))
                if opts:
                    pid = col.selectbox(role.capitalize(), opts, key=f"sim_lineup_{key}_{who}_{role}",
                                        format_func=lambda p, label=label: label.get(p, sim.player_names.get(p, p)))
                    chosen.append((role, pid))
            lineups[who] = tuple(chosen)
    setup = S.Setup(our_side=our_side, we_first=first == "us", game_number=game_no, locked=frozenset(locked),
                    lineup_us=lineups["us"], lineup_them=lineups["them"])

    # Replay the stored draft against the current settings; cut it where it became illegal.
    acts = ()
    for a in st.session_state.get(skey, ()):
        try:
            acts = sim.apply(setup, acts, a.champion_id, a.role)
        except ValueError as e:
            st.warning(f"Draft cut back to before {champ_names[a.champion_id]}: {e}")
            break
    st.session_state[skey] = acts
    if err := st.session_state.pop(skey + "_err", None):
        st.error(err)

    board = S.slots(setup)
    nxt = board[len(acts)] if len(acts) < S.N_SLOTS else None
    by_slot = {a.slot: a for a in acts}

    # --- board ---
    for col, side in zip(st.columns(2), ("blue", "red")):
        who = "us" if side == our_side else "them"
        lineup = sim.lineup(setup, who)
        with col:
            first_txt = " · first pick" if (who == "us") == setup.we_first else ""
            st.markdown(f"#### :{side}[{side.capitalize()}] · {team_label[who]}{first_txt}")
            chips = []
            for s in (s for s in board if s["side"] == side and s["kind"] == "ban"):
                a = by_slot.get(s["slot"])
                is_next = nxt is not None and nxt["slot"] == s["slot"]
                chips.append(chip(champ_names[a.champion_id] if a else s["label"],
                                  "chip-next" if is_next else ("" if a else "chip-empty")))
            st.markdown("Bans " + "".join(chips), unsafe_allow_html=True)
            for s in (s for s in board if s["side"] == side and s["kind"] == "pick"):
                a = by_slot.get(s["slot"])
                is_next = nxt is not None and nxt["slot"] == s["slot"]
                c1, c2 = st.columns([1.7, 1.3], vertical_alignment="center")
                label = f"<b>▶ {s['label']}</b>" if is_next else s["label"]
                if a:
                    player = html.escape(sim.player_names.get(lineup.get(a.role), ""))
                    c1.markdown(f"<span style='white-space:nowrap'>{label}</span> · "
                                f"<b>{html.escape(champ_names[a.champion_id])}</b> · {player}", unsafe_allow_html=True)
                    wkey = f"sim_role_{key}_{s['slot']}_{a.champion_id}_{a.role}"
                    c2.selectbox("Role", S.ROLES, index=S.ROLES.index(a.role), key=wkey, label_visibility="collapsed",
                                 format_func=ROLE_SHORT.get,
                                 on_change=sim_set_role, args=(skey, sim, setup, s["slot"], wkey))
                else:
                    c1.markdown(f"<span style='white-space:nowrap'>{label}</span> · <span style='color:#9aa0a6'>–</span>",
                                unsafe_allow_html=True)

    # --- next action ---
    if nxt is None:
        st.success("Draft complete.")
    else:
        st.markdown(f"**Next: {nxt['label']} · {team_label[nxt['who']]} ({nxt['side']})**")
    a1, a2 = st.columns([3, 2], vertical_alignment="bottom")
    a3, a4, a5, _ = st.columns([1.3, 1, 1, 2])
    champ, role = None, None
    if nxt is not None:
        gone = sim.unavailable(setup, acts)
        champ = a1.selectbox("Champion", [c for c in champ_names.index if c not in gone], index=None,
                             placeholder="Type to search...", format_func=champ_names.get,
                             key=f"sim_choice_{key}_{len(acts)}")
        if nxt["kind"] == "pick":
            open_r = sim.open_roles(setup, acts, nxt["who"])
            guess = sim.guess_role(setup, acts, nxt["who"], champ) if champ else open_r[0]
            role = a2.selectbox("Role", open_r, index=open_r.index(guess), key=f"sim_newrole_{key}_{len(acts)}_{champ}",
                                help="Defaults to the role this team usually plays the champion in.")
        a3.button("Lock in", type="primary", disabled=champ is None, on_click=sim_apply,
                  args=(skey, sim, setup, champ, role), key=f"sim_lock_{key}")
    a4.button("Undo", disabled=not acts, on_click=sim_undo, args=(skey,), key=f"sim_undo_{key}")
    a5.button("Reset", disabled=not acts, on_click=sim_reset, args=(skey,), key=f"sim_reset_{key}")

    # --- win chance ---
    st.divider()
    wc, traj = sim.win_chance(setup, acts), sim.trajectory(setup, acts)
    m1, m2 = st.columns([1, 2.4])
    m1.metric("Our win chance", f"{wc['p']:.0%}", f"90% range {wc['lo']:.0%}–{wc['hi']:.0%}",
              delta_color="off", delta_arrow="off",
              help=f"Range = spread across {wc['refits']} refits of the model on resampled series: how sure the "
                   f"model is of its own estimate. Any single game is far less certain.")
    m1.metric("Team strength only", f"{wc['p_elo']:.0%}", f"Elo {wc['elo_us']:.0f} vs {wc['elo_them']:.0f}",
              delta_color="off", delta_arrow="off")
    fig = go.Figure([
        go.Scatter(x=traj["step"], y=traj["hi"] * 100, mode="lines", line={"width": 0}, hoverinfo="skip",
                   showlegend=False),
        go.Scatter(x=traj["step"], y=traj["lo"] * 100, mode="lines", line={"width": 0}, fill="tonexty",
                   fillcolor="rgba(76,120,168,0.2)", name="90% range", hoverinfo="skip"),
        go.Scatter(x=traj["step"], y=traj["p"] * 100, mode="lines+markers", line={"color": "#4C78A8"},
                   name="Win chance", customdata=traj["label"],
                   hovertemplate="%{customdata}<br>%{y:.1f}%<extra></extra>"),
    ])
    fig.add_hline(y=50, line={"color": GREY, "dash": "dot", "width": 1})
    fig.update_layout(height=260, margin={"l": 10, "r": 10, "t": 10, "b": 10}, showlegend=False,
                      yaxis={"title": f"{our_name} win %", "range": [max(0, traj["lo"].min() * 100 - 5),
                                                                       min(100, traj["hi"].max() * 100 + 5)]},
                      xaxis={"tickmode": "array", "tickvals": traj["step"], "ticktext": traj["short"],
                             "range": [-0.5, S.N_SLOTS + 0.5]})
    m2.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("**The draft typically shifts odds by about 3 points; team strength matters more.** In testing on "
               "897 later games the draft added no reliable accuracy beyond team strength (docs/MODEL.md). "
               "Bans don't move this line directly; their value shows in what they deny. Always taking the top "
               "option below overstates the gain, because the best of many noisy estimates is usually too high.")

    # --- best options / likely from them ---
    o1, o2 = st.columns(2)
    with o1:
        st.markdown(f"**Best options for {our_name}**")
        best = sim.best_options(setup, acts)
        if nxt is None:
            st.caption("Draft complete.")
        elif best is None:
            st.caption(f"{opp_name}'s turn: see their likely choices.")
        elif best.empty:
            st.caption("No candidates for the open roles.")
        else:
            if nxt["kind"] == "pick":
                st.caption("Points = change in our win chance if we pick it now, with the 90% range across model refits.")
            else:
                st.caption("Points = chance they pick it later in this draft × how much worse it is for us than their "
                           "typical pick. Role = theirs.")
            for r in best.itertuples():
                st.button(f"{nxt['kind'].capitalize()} {r.champion}", key=f"sim_opt_{key}_{len(acts)}_{r.champion_id}",
                          on_click=sim_apply,
                          args=(skey, sim, setup, r.champion_id, r.role if nxt["kind"] == "pick" else None))
                st.markdown(f"**{r.shift_pts:+.1f}** pts ({r.lo_pts:+.1f} to {r.hi_pts:+.1f}) · {r.role} · {r.reason}"
                            f"<div class='opt-ev'>{html.escape(r.evidence)}</div>", unsafe_allow_html=True)
    with o2:
        ln = sim.likely_next(setup, acts)
        if ln is None:
            st.markdown(f"**Likely from {opp_name}**")
            st.caption("No more actions for them.")
        else:
            when = "now" if nxt is not None and nxt["slot"] == ln["slot"]["slot"] else "their next turn"
            st.markdown(f"**Likely from {opp_name}: {ln['slot']['label']} ({when})**")
            t = ln["table"]
            st.dataframe(t.assign(prob=t["prob"] * 100), hide_index=True, width="stretch",
                         column_order=["champion"] + (["role"] if ln["kind"] == "pick" else []) + ["prob", "evidence"],
                         column_config={"champion": "Champion", "role": "Role",
                                        "prob": st.column_config.ProgressColumn("Chance", format="%.0f%%",
                                                                                min_value=0, max_value=100),
                                        "evidence": st.column_config.TextColumn("Evidence", width="large")})
            acc = likely_accuracy().set_index(["kind", "method"])
            ours_acc, meta_acc = acc.loc[(ln["kind"], "tendency")], acc.loc[(ln["kind"], "meta only")]
            st.caption(f"From their habits by draft phase over the last year, the league meta, their open roles and "
                       f"what's still available. Checked on {int(ours_acc['predictions']):,} past {ln['kind']}s in "
                       f"{int(ours_acc['games'])} games, each using only earlier games: the actual {ln['kind']} was in "
                       f"this top 5 {ours_acc['top5']:.0%} of the time (league meta alone: {meta_acc['top5']:.0%}).")


# --- tab: insights --------------------------------------------------------------------------
@st.cache_data(show_spinner="Checking which kinds of pattern hold up in later games (first run: about 30 s)...")
def insight_replication() -> pd.DataFrame:
    return I.load_replication(get_scout())


@st.cache_data(show_spinner="Writing insight cards...")
def insight_report(team, patches, view):
    return get_scout().report(team, patches, view, replication=insight_replication())


def insight_card(c) -> None:
    with st.container(border=True):
        st.markdown(f"**{html.escape(c.title)}**")
        st.markdown(c.takeaway)
        for line in c.as_dict()["evidence"]:
            st.caption(line)
        st.caption(f"Confidence {c.confidence:.0%} · this kind of pattern kept {c.reliability:.0%} of its gap in "
                   f"later games · {c.games} {c.unit}",
                   help="Confidence = 1 − p-value: how unlikely a gap this big is if there were no real difference. "
                        "'Kept X% of its gap' comes from checking the same kind of pattern on later games for "
                        "every team (see 'Are these patterns real?').")


def insight_list(rep: dict, empty_msg: str) -> None:
    if not rep["top"]:
        st.info(empty_msg)
    for c in rep["top"]:
        insight_card(c)
    if 0 < rep["passing"] < 5:
        st.caption(f"Only {rep['passing']} pattern{'s' if rep['passing'] != 1 else ''} passed the bar "
                   f"(confidence ≥ {I.MIN_CONFIDENCE:.0%}, and a kind of pattern that held up in later games).")


def all_patterns(rep: dict) -> None:
    rows = pd.DataFrame([c.as_dict() for c in rep["cards"]])
    if rows.empty:
        return
    with st.expander(f"All {len(rows)} patterns found, including the ones that didn't hold up"):
        st.caption("Side, first-pick and Fearless carry-over gaps did not hold up in later games, and most "
                   "low-confidence ban gaps were chance (see 'Are these patterns real?'): read those rows as "
                   "anecdotes, not plans.")
        st.dataframe(rows.assign(effect=rows["effect"] * 100, confidence=rows["confidence"] * 100,
                                 reliability=rows["reliability"] * 100,
                                 sample=rows["games"].astype(str) + " " + rows["unit"]),
                     hide_index=True, width="stretch",
                     column_order=["kind", "title", "effect", "confidence", "reliability", "sample", "takeaway"],
                     column_config={"kind": "Kind", "title": "Pattern",
                                    "effect": st.column_config.NumberColumn("Gap (pts)", format="%.0f"),
                                    "confidence": st.column_config.NumberColumn("Confidence", format="%.0f%%"),
                                    "reliability": st.column_config.NumberColumn("Held up later", format="%.0f%%"),
                                    "sample": "Sample", "takeaway": st.column_config.TextColumn("Takeaway", width="large")})


with tab_insights:
    what_this_means(f"plain-language counter-strategies against {opp_name}, built from their own records and draft "
                    f"habits (not from the win model). Each one shows its evidence, a confidence level and the "
                    f"sample behind it. Only patterns of a kind that held up in later games make the list.")
    ins_patches = st.slider("Window: each team's last N patches", 4, info["n_patches"],
                            min(I.DEFAULT_PATCHES, info["n_patches"]), key="insights_patches",
                            help="Insights need more games than the other tabs: the default is 12 patches.")
    opp_rep = insight_report(opp_id, ins_patches, "opponent")
    st.subheader(f"How to beat {opp_name}")
    st.caption(f"From {opp_rep['games']} {opp_name} games. Ranked by how large the pattern is for its kind × how "
               f"confident it is × how much of that kind of pattern held up in later games.")
    insight_list(opp_rep, f"No pattern passes the bar for {opp_name} in this window. Try a wider window.")
    st.markdown(f"**{opp_name}'s draft habits**")
    for line in opp_rep["predictability_takeaways"]:
        st.markdown(f"- {line}")
    all_patterns(opp_rep)

    st.subheader("Are these patterns real?")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Their habits predict their next pick**")
        acc = likely_accuracy()
        st.dataframe(acc.assign(top1=acc["top1"] * 100, top5=acc["top5"] * 100), hide_index=True, width="stretch",
                     column_order=["kind", "method", "predictions", "top5", "top1"],
                     column_config={"kind": "Next", "method": "Guess from",
                                    "predictions": st.column_config.NumberColumn("Checked", format="%d"),
                                    "top5": st.column_config.NumberColumn("Actual in top 5", format="%.0f%%"),
                                    "top1": st.column_config.NumberColumn("Top guess right", format="%.0f%%")})
        st.caption(f"Every pick and ban of {int(acc['games'].iloc[0])} games from July 2025 on, each guessed from "
                   f"earlier games only. 'tendency' = the team's own habits by draft phase plus the meta; "
                   f"'meta only' = what the whole league picks. Habits beat the meta, so tendencies are real.")
    with r2:
        st.markdown("**Which kinds of pattern hold up later**")
        rep_tab = insight_replication()
        st.dataframe(rep_tab.assign(same_direction=rep_tab["same_direction"] * 100, survives=rep_tab["survives"] * 100,
                                    early_gap=rep_tab["early_gap"] * 100, late_gap=rep_tab["late_gap"] * 100,
                                    all_same_direction=rep_tab["all_same_direction"] * 100),
                     hide_index=True, width="stretch",
                     column_order=["kind", "checks", "same_direction", "early_gap", "late_gap", "survives",
                                   "all_same_direction"],
                     column_config={"kind": "Kind", "checks": st.column_config.NumberColumn(
                                        "Checks", help=f"Patterns with confidence ≥ {I.MIN_CONFIDENCE:.0%}"),
                                    "same_direction": st.column_config.NumberColumn("Same direction", format="%.0f%%"),
                                    "all_same_direction": st.column_config.NumberColumn(
                                        "All patterns: same dir.", format="%.0f%%",
                                        help="Including the low-confidence ones"),
                                    "early_gap": st.column_config.NumberColumn("Gap before", format="%.1f"),
                                    "late_gap": st.column_config.NumberColumn("Gap after", format="%.1f"),
                                    "survives": st.column_config.NumberColumn("Kept", format="%.0f%%")})
        st.caption("For every team, confident patterns found in games before a split date (July 2025 and April "
                   "2026) were measured again on later games. 50% same direction = coin flip. Draft habits held up "
                   "best; comfort dependency and confident ban-impact gaps partly; side, first pick and Fearless "
                   "carry-over gaps did not. Most low-confidence ban gaps were chance.")

    st.subheader(f"Self-scouting: {our_name}")
    self_rep = insight_report(our_id, ins_patches, "self")
    st.caption(f"The same analysis on {our_name} ({self_rep['games']} games): what an opponent's analyst could exploit.")
    insight_list(self_rep, f"No pattern passes the bar for {our_name} in this window.")
    st.markdown("**How predictable are we?**")
    for line in self_rep["predictability_takeaways"]:
        st.markdown(f"- {line}")
    pred = self_rep["predictability"]
    p1, p2 = st.columns(2)
    for col, kind, label in ((p1, "pick", "First pick"), (p2, "ban", "First ban")):
        x = pred[kind]
        col.metric(f"{label}: effective choices", f"{x['effective_choices']:.1f}",
                   f"{x['distinct']} different in {x['games']} games", delta_color="off", delta_arrow="off",
                   help="2^entropy (Miller–Madow corrected): the number of equally likely options that would be "
                        "as hard to predict. Higher = harder to prepare for.")
    all_patterns(self_rep)


# --- tab: match prep (first tab; built last so it can reuse every cached loader above) ------
@st.cache_data(show_spinner="Preparing the match brief...")
def match_prep(our, opp, weights_items):
    return P.match_prep(our, opp, drafts=get_drafts(), scout=get_scout(), sim=get_sim(our, opp),
                        insight_rep=insight_replication(), likely_acc=likely_accuracy(), weights=dict(weights_items))


def badge(text: str) -> str:
    return f'<span class="badge">{html.escape(text)}</span>'


def details(*tabs: str) -> None:
    st.caption("Details: " + " · ".join(f"**{t}** tab" for t in tabs))


with tab_prep:
    what_this_means(f"the 60-second brief before playing {opp_name}: what they'll draft, what to ban, how to beat "
                    f"them and what they'll try against us. Each block names the tab with the details.")
    p = match_prep(our_id, opp_id, tuple(sorted(weights.items())))
    top_l, top_r = st.columns([3, 1], vertical_alignment="bottom")
    top_l.caption(f"Records, form and bans: each team's last {p['window']} patches · insights: last "
                  f"{p['insights_window']} patches · data through {p['data_through']}.")
    top_r.download_button("Export pre-match report", data=P.render_html(p), mime="text/html",
                          file_name=f"prematch_{our_name}_vs_{opp_name}.html".replace(" ", "_"),
                          help="A one-page, text-only HTML report. Open it in a browser and print to PDF.")

    h2h, fu, ft = p["h2h"], p["form"]["us"], p["form"]["them"]
    m1, m2, m3 = st.columns(3)
    m1.metric("Head-to-head", P.record_text(h2h["window"]),
              f"{h2h['window']['games']} games in window · {P.record_text(h2h['all'])} all-time",
              delta_color="off", delta_arrow="off")
    for col, f, who in ((m2, fu, our_name), (m3, ft, opp_name)):
        col.metric(f"{who} form", f"{f['wins']}–{f['losses']}",
                   f"last {f['last_n']}: {' '.join(f['last']) or '–'}", delta_color="off", delta_arrow="off")
    details("Opponent scouting")

    left, right = st.columns(2)
    with left:
        st.markdown(f"#### What {opp_name} will draft")
        lk = pd.DataFrame(p["likely"])
        if lk.empty:
            st.info("No games to learn their habits from.")
        else:
            st.dataframe(lk.assign(prob=lk["prob"] * 100), hide_index=True, width="stretch",
                         column_order=["champion", "role", "prob", "evidence"],
                         column_config={"champion": "Champion", "role": "Role",
                                        "prob": st.column_config.ProgressColumn("Chance", format="%.0f%%",
                                                                                min_value=0, max_value=20),
                                        "evidence": st.column_config.TextColumn("Evidence", width="large")})
            st.caption("Most likely first-phase picks.")
        for t in p["tells"]:
            st.markdown(f"- {html.escape(t['takeaway'])} {badge(f'{t['n']} {t['unit']}')}", unsafe_allow_html=True)
        details("Draft simulator", "Insights")
    with right:
        st.markdown("#### Our ban plan")
        if not p["bans"]:
            st.info(f"No champion qualifies: {opp_name} has too few games in the window.")
        for i, b in enumerate(p["bans"], 1):
            st.markdown(f"{i}. **{html.escape(b['champion'])}**: {html.escape(b['lead'])}"
                        f"<div class='opt-ev'>{html.escape(' · '.join(b['more']))}</div>", unsafe_allow_html=True)
        st.caption(f"From {p['ban_games']} {opp_name} games, strongest reason first. {p['ban_proof']}")
        details("Ban plan")

    st.markdown(f"#### How to beat {opp_name}")
    if not p["beat"]:
        st.info(f"No insight card passes the bar for {opp_name} in this window.")
    else:
        for col, c in zip(st.columns(len(p["beat"])), p["beat"]):
            with col.container(border=True):
                st.markdown(f"**{html.escape(c['title'])}**")
                st.markdown(c["takeaway"])
                st.markdown(badge(f"{c['n']} {c['unit']} · confidence {c['confidence']:.0%}"), unsafe_allow_html=True)
    details("Insights")

    w1, w2 = st.columns(2)
    with w1:
        st.markdown(f"#### Watch out: what {opp_name} bans against us")
        if not p["they_ban"]:
            st.info(f"No head-to-head games between {our_name} and {opp_name} in the data.")
        for x in p["they_ban"]:
            st.markdown(f"- **{html.escape(x['champion'])}**: banned in {x['bans']} of {x['games']} games "
                        + (badge("small sample") if x["small_sample"] else ""), unsafe_allow_html=True)
        details("Against us")
    with w2:
        st.markdown("#### Watch out: our predictable habits")
        for line in p["habits"]:
            st.markdown(f"- {line}")
        details("Insights (self-scouting)")
    st.info(p["proof"])


# --- footer --------------------------------------------------------------------------------
st.divider()
st.caption(
    f"**Roles are inferred**: the data has no role field. Roles come from roster order in early-2025 events and "
    f"from each player's usual role and champion elsewhere (100% accurate on held-out labelled games). "
    f"**Data**: GRID official match data (Sky's The Limit 2026 hackathon), {info['games']} pro games, "
    f"patches {info['first_patch']}–{info['last_patch']} ({info['first_date']} to {info['last_date']}). "
    f"This view: {span_text.lower()}. Text only; no game artwork is used."
)
