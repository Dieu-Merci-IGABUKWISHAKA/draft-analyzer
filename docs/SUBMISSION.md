# Devpost submission (ready to paste)

## Project name
Draft Tendency Analyzer

## Tagline
Draft prep for Cloud9's coaches from GRID data: what the opponent will draft, what to ban and how to beat them, showing only advice that held up on games it had never seen.

## Inspiration
I'm a final-year Electrical Engineering student from Rwanda, doing the ALX Data Science programme, and this is my first esports hackathon. Draft prep is hours of watching VODs and tallying picks by hand. The data that answers "what will they play, and what should we take away?" already exists in GRID's event feeds. I wanted Cloud9's coaches to get that answer in seconds. I also wanted every answer to show its evidence, because a 4–0 record on a champion is not 40 games of proof, and a pattern that looks strong this month may not repeat next month.

## What it does
You pick our team (Cloud9 Kia by default) and an opponent. The dashboard has seven tabs:
- **Match prep:** the 60-second brief. It covers head-to-head and form, what they'll draft, our top 5 bans, the top 3 "how to beat them" cards, and what to watch out for. It exports to a one-page printable report.
- **Draft simulator:** you play a 20-slot pro draft pick by pick. First pick and side are set separately (2026 "first selection"), along with the game number and Fearless-locked champions. Each step shows our win chance with a 90% range next to team strength alone, the best 5 options with what each shifts, and the opponent's 5 most likely next picks or bans.
- **Opponent scouting:** champion pool by inferred role, priority picks, bans by phase, and early-game objectives (only the patterns that held up).
- **Insights:** "How to beat <opponent>" cards with Wilson 90% intervals, confidence and sample size, plus self-scouting of our own predictable habits.
- **Ban plan:** the five bans that hurt them most, each led by its strongest plain-English reason ("Picked 7 of 39 games · Win rate 86% (6–1) · Banned vs them in 4 games").
- **Fearless series:** their record, champion variety and top picks in games 1 to 5, plus a series planner. You mark the champions already used and get the next game's bans.
- **Against us:** what they ban against us, and what we ban against them.

Every rate shows its sample size, and small samples are flagged.

## How I built it
I set the goal and the direction, planned with Claude in chat, and had Claude write and test the code inside PyCharm. I checked each result before building on it.
- **Pipeline:** Python streams all 745 LoL series (zstd JSON Lines) and takes each game's final draft snapshot. It joins patch, sides and winner from GRID's metadata on IDs, and maps every action to its standard draft slot (1–20). The same pass collects first blood, dragon, void grub, herald and tower events.
- **Roles** are inferred, because the data has none.
- **Models:** an Elo + draft win-probability model (scikit-learn logistic regression), opponent-habit tables for likely picks, and a ban score you can read.
- **Validation:** everything is fitted on earlier games and tested on later ones (walk-forward by month or by date), never on a random split.
- **UI:** all logic lives in one Python package. A Streamlit + Plotly dashboard only lays it out.
- **Tools:** PyCharm with JetBrains Air (the in-IDE agent, running Claude), shared run configurations, ruff, pytest and GitHub Actions.

## Challenges I ran into
- **Choosing a draft source.** The event stream and the final snapshot disagree in 5 games, and the stream carries 461 undo events.
- **2026 first selection.** The first-pick team can be on red side, which broke my blue-first draft template.
- **No role field.** Roster order only matches roles in early-2025 events.
- **The draft barely predicts the winner.** My win model looked promising on one split, then showed no reliable gain over team strength on 897 later games. I kept the honest result and reframed the simulator around "what each choice shifts".
- **Most patterns don't repeat.** Win-rate gaps by side, first pick or Fearless carry-over, and most early-game habits, were coin flips in later games, so I stopped showing them as advice.

## Accomplishments that I'm proud of
- **Ban plan, tested on 439 games from 2026 it never saw:** the opponent picked a listed champion 34% of the time when it was left open, vs 27% for the league meta's top 5 and 20% for their own top 5.
- **Likely picks, tested on 897 games:** their actual next pick was in our top 5 44% of the time, vs 37% for the meta.
- **Only advice that held up is shown:** draft habits pointed the same way in later games 82% of the time, so they qualify. The kinds that didn't hold up are labelled as such.
- **Fearless Draft detected from the data:** no champion is repeated in 568 multi-game series, across all 20 events.
- **Role inference** correct on 100% of held-out labelled games.
- **License-safe throughout:** no data in the repo or its history.

## What I learned
- How pro LoL drafting works (phases, first pick, Fearless), and how much 2026's first-selection rule changes draft analysis.
- Testing on later games changes the product: more than half of my first ideas didn't survive it.
- Small samples need to be visible, smoothed and filtered, or they dominate any ranking.
- A score a coach can read, with weights they can adjust, earns more trust than a black box.

## What's next
- Counter-pick suggestions by lane matchup, with enough games behind them.
- Team-composition tendencies (engage, poke, scaling) on top of single champions.
- Live input from an ongoing series instead of manual entry.
- VALORANT map and agent scouting with the same approach.

## Methodology write-up (~250 words)
I use the final `seriesState.games[].draftActions` of each GRID feed as the source of truth. It matched the event stream in 1,734 of 1,739 games, held bans the stream missed in the other 5, and already resolves undo events.
- **Draft slots** are keyed on the team that picks first, which is red in 243 of the 2026 games.
- **Roles** come from 6 early-2025 events whose roster order is top-to-support, then a best one-player-per-role assignment elsewhere (100% on held-out events).
- **Recency:** a game is weighted 0.5^(patches ago / 3). Win rates get 10 imaginary games at the team's average.
- **Ban score:** recent pick rate × (2 + 3 × smoothed win-rate edge over 50%), plus how often other teams ban it against them. Weights were tuned on 2025 and tested on 2026 (34% vs 27% meta).
- **Win model:** Elo, side and draft features (meta, comfort, matchups), each from earlier dates only. It is trained at every draft stage and tested walk-forward by month on 897 games: 65.2% accuracy vs 64.1% for Elo alone, a gain that isn't statistically real.
- **Insights:** each kind of pattern is re-measured on games after two split dates. Only kinds that kept their direction and part of their gap qualify as advice.
- **Fearless** is detected from repeated picks (none in 568 series).
- **Limitations:** small samples, inferred roles, associations not causes, and a few excluded games. All are documented in docs/VALIDATION.md.

## Tooling explanation (~100 words)
- **PyCharm:** the project, the Python 3.13 interpreter, git, the pytest runner and seven shared run configurations (dashboard, rebuild, model, backtest, tests, setup check, scouting report). `.idea/` is committed.
- **JetBrains Air**, the agent panel in PyCharm running Claude, wrote and tested Phases 0–8 from a written project brief (AGENTS.md). Claude Code in PyCharm's terminal did the initial setup, and I planned with Claude in chat.
- **Me:** I set the questions and rules, made the judgment calls and checked every number.
- **Quality:** ruff and pytest, run by GitHub Actions on every push.
- **Data:** GRID's official data, streamed from compressed files and never committed.

## Built with
python, pandas, numpy, scikit-learn, streamlit, plotly, pycharm, jetbrains-air, claude, grid
