# Devpost submission (ready to paste)

## Project name
Draft Tendency Analyzer

## Tagline
Scout any LoL pro team's picks and bans from GRID data, get 5 ban recommendations with reasons and sample sizes, and re-plan bans mid-series under Fearless Draft.

## Inspiration
Draft prep is hours of watching VODs and tallying picks by hand, and the data that answers "what will they play, and what should we take away?" already exists in GRID's event feeds. We wanted Cloud9's coaches to get that answer in seconds. We also wanted the answer to show its evidence, because a 4–0 record on a champion is not the same as 40 games of proof.

## What it does
Pick our team (Cloud9 Kia by default) and an opponent. The dashboard shows:
- **Opponent scouting:** their champion pool by role, their priority picks with and without first pick, and their bans in phase 1 vs phase 2.
- **Against us:** what they ban against Cloud9, and what Cloud9 bans against them.
- **Ban plan:** the five bans that hurt them most, each with a plain-English reason ("Picked 7 of 39 games · Win rate 86% (6–1) · Banned vs them in 4 games") and a small-sample warning where it applies.
- **Fearless series:** how they draft and win in game 1, game 2 and game 3+, plus a series planner. Mark the champions already used, and it shows what the opponent loses, what they'll likely pick now, and a new ban plan.

Recency (last N patches, half-life) and every ban-score weight are adjustable.

## How we built it
- **Pipeline:** a Python pipeline streams all 745 LoL series (zstd JSON Lines) and takes each game's final draft snapshot. It joins patch, sides and winner from GRID's metadata on IDs, and maps every action to its standard draft slot (1–20).
- **Roles:** inferred, because the data has none.
- **Analysis and UI:** all metrics live in one analysis module with pandas; a Streamlit + Plotly dashboard only displays them.
- **Tools:** PyCharm, with Claude Code run in its terminal for coding.

## Challenges we ran into
- **Choosing a draft source:** the event stream and the final snapshot disagree in 5 games, and the stream carries 461 undo events.
- **2026 first selection:** the first-pick team can be on red side, which broke our blue-first draft template.
- **No role field:** roster order only matches roles in early-2025 events.
- **Ban score:** our first version ranked a 44%-win-rate champion and a 4–0 sample too high, and needed reworking.
- **Bad series:** one series was overwritten by an invalidated replay.

## Accomplishments that we're proud of
- **Sources compared on all 745 series**, not a sample, before picking one.
- **Role inference** correct on 100% of held-out labelled games.
- **Fearless Draft detected from the data:** no champion is repeated in 568 multi-game series, across all 20 events.
- **Every number shows its sample size.**
- **License-safe throughout:** no data in the repo or its history.

## What we learned
- Pro LoL drafting (phases, first pick, Fearless) and how much 2026's first-selection rule changes draft analysis.
- Small samples need to be visible, smoothed and filtered, or they dominate any ranking.
- A transparent score with adjustable weights earns more trust from a coach than a black box.

## What's next
- Pick recommendations and counter-picks by role, not only bans.
- Player-level champion pools and team-composition tendencies.
- Comparing draft prep against actual series outcomes to tune the ban weights.
- VALORANT map and agent scouting with the same approach.

## Methodology write-up (~250 words)
We use the final `seriesState.games[].draftActions` of each GRID feed as the source of truth.
- **Source choice:** compared with the `team-picked/banned-character` event stream across all 745 series, the snapshot matched in 1,734 of 1,739 games. It also held bans the stream missed in the other 5, and it already resolves "!" undo events and rolled-back games.
- **Draft slots:** each action is mapped to its standard pro draft slot (6 bans, 6 picks, 4 bans, 4 picks), keyed on the team that picks first. That team is red in 243 of the 2026 games.
- **Inferred roles:** roles come from 6 early-2025 events whose roster order is top-to-support. Those labels give each champion's and each player's role tendencies, and every other game gets the best one-player-per-role assignment (100% accuracy on held-out labelled events).
- **Recency:** a game is weighted 0.5^(patches ago / 3) within a last-N-patches window.
- **Smoothing:** win rates get 10 imaginary games at the team's average.
- **Ban score:** recent pick rate × (2 + 3 × (smoothed win rate − 50%) / 50%), plus 1 × how often other teams ban it against them. Champions under 2 picks are not ranked. The weights were tuned on 2025 games and backtested on 2026 games: the opponent picked 33.8% of the listed champions left open, against 19.8% for their own top 5 and 26.9% for the league meta (docs/VALIDATION.md).
- **Fearless detection:** no champion is ever re-picked within any of 568 multi-game series, so every event is hard Fearless. The series planner removes used champions for both teams.
- **Limitations:** small samples (flagged everywhere), inferred roles, and a handful of games with missing or invalidated drafts, which are excluded and documented.

## Tooling explanation (~100 words)
- **PyCharm:** the project, the Python 3.13 interpreter, git, and shared run configurations for the dashboard, the data rebuild and the scouting report. The `.idea/` folder is committed as evidence.
- **Claude Code**, run in PyCharm's terminal, wrote and ran the code. A person decided the scouting questions, made the judgment calls (how to treat roles, how the ban score should behave) and checked each result against the data.
- **Data:** GRID's official match data, streamed directly from compressed files and never committed.
- **Stack:** pandas and pyarrow for analysis, Streamlit and Plotly for the coach-facing dashboard.

## Built with
python, pandas, streamlit, plotly, pycharm, grid
