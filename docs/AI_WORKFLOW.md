# How I built this with AI tools

I'm a final-year Electrical Engineering student from Rwanda, and I'm doing the ALX Data Science programme. Sky's The Limit is my first esports hackathon. I chose it, and I chose Category 2 (LoL macro scouting and draft management): helping a coach prepare the draft against one specific opponent.

I did not write most of the code by hand. I directed AI tools, made the decisions and checked the results. This page explains how that worked, including the moments where I pushed back or changed course.

## Who did what

- **Me.** I set the goal (win Category 2), decided which questions mattered to a coach, set the rules, chose the direction at each step and checked every result before building on it.
- **Claude in chat.** I planned with it: what to build, in what order, and how to test it.
- **Claude Code in PyCharm's terminal.** It did the first setup: data checks, the drafts table, roles and a first dashboard.
- **Claude in JetBrains Air**, the agent panel inside PyCharm. It wrote and tested the code for Phases 0–8 and reported back with numbers I could check.

Most of it was built on 20 November 2026, from about 15:00 to 23:30 (see the git log). I did a docs and screenshots pass the next day. Setup ran from 15:00 to 16:40 in Claude Code, and Phases 0–8 ran from 17:00 onward in JetBrains Air.

## The main change of course: from a dashboard to a tested tool

By 16:40 I had a working dashboard and a first set of submission docs. At that point I decided to aim higher: not a basic dashboard, but a tool whose advice is tested on games it has never seen.

That became Phase 0: a written project brief, [AGENTS.md](../AGENTS.md). It holds the judges' questions, the hard rules and the current findings. The rules are that the licensed data is never committed, that every number shows its sample size, and that anything fitted or tuned is checked on later games only. Every task after that started from the brief.

## Where I pushed back

- **Roles.** The drafts have no role field. I told the agent to tell me before inventing one, and it stopped to ask. Its first idea was to use roster order, but that turned out to be role order only in early-2025 events: Cloud9's 2026 list started with their support. So it switched to learning roles from the labelled events and inferring the rest. It reported a held-out accuracy of 100% before we used them.
- **The ban score.** Version 1 ranked Nocturne #2 against LYON, even though LYON wins only 44% on it, because "other teams ban it" weighed too much. I asked for that weight to drop and for losing records to count against a ban. Version 2 then put a 3–0 LeBlanc and a 4–0 Varus on top. I asked for stronger smoothing and a minimum-picks rule, until Ryze (6–1 in 7 games) outranked Varus (4–0).
- **Only showing advice that held up.** Once the checks on later games came back, many patterns turned out to be coin flips. I decided the dashboard would show as advice only the kinds of pattern that held up, and label the rest as such, instead of showing everything that looked confident.
- **Keeping the negative results.** The win model added nothing reliable beyond team strength. I kept that result visible, and the simulator was reframed around "what each choice shifts", instead of hiding the result or calling the estimate a forecast.

## Where the AI did the heavy lifting

- **License-safe setup.** The data was cloned sparsely next to the project, and the `.gitignore` blocks every data format. The agent ran the data check before every commit, and scanned the full history before the first public push.
- **Choosing the data source with evidence.** Instead of sampling 2–3 series, the agent compared the event stream with the final snapshot across all 745 series. They matched in 1,734 of 1,739 games. The stream missed bans in the other 5 and carried 461 undo events, so we went with the snapshot.
- **Finding the 2026 first-selection change.** The first draft-slot mapping assumed blue side picks first, and over 200 games didn't fit. The agent traced them all to 2026, where the first-pick team is often red, and re-keyed the slots on "the team that picks first".
- **Detecting Fearless from the data.** Instead of hard-coding which leagues use Fearless Draft, it checked for repeated champions within each series. There were none in 568 multi-game series across all 20 events.
- **Honesty checks.** Before writing the first docs, the agent re-checked its claims. It flagged a library we weren't using (duckdb) and one of its own earlier counts that was wrong (663 instead of 568 series).
- **Walk-forward validation everywhere.** The win model, the likely-picks table, the insight cards, the early-game patterns and the Ban plan are all fitted on earlier games and checked on later ones. The agent built each of those checks, and every result is collected in [VALIDATION.md](VALIDATION.md).

## What the checks changed (Phases 1–8)

Most of the product's shape came from results that didn't hold up, not from the ones that did.

| Phase | What we expected | What the check on later games showed | What changed |
|---|---|---|---|
| 1–2. Win model | The draft predicts the winner | A single split looked good (+0.011 log loss on 275 games). The monthly walk-forward test on 897 games showed no reliable gain over Elo. | The simulator shows "what each choice shifts", never a forecast. |
| 2. Simulator | Taking the top option each turn is a fair test | In a test draft vs LYON, always taking the top option pushed the estimate from 40% to 51% (winner's curse: the best of many noisy estimates is too high). | The simulator warns about it on screen. |
| 3. Insights | Every confident pattern is advice | Draft habits held up (82% same direction). Side, first-pick and Fearless carry-over gaps were coin flips. | Only kinds of pattern that held up can reach "How to beat". |
| 4. Early game | Objective habits are scoutable | Only first-tower rate and the jungler–first-grub link held up. | The tab lists those two; everything else is in an expander. |
| 5. Ban plan | The hand-set weights work | They were no better than the league meta in 2026 (26.5% vs 26.9%). | Weights tuned on 2025 became the defaults: 33.8% vs 26.9% on 2026. |
| 6. Product pass | (Browser checks for four opponents and at 600 px width) | Chart toolbars covered legends; "1 games" labels. | Fixed, and Match prep became the first tab. |
| 7. Quality | | | ruff, type hints, pytest in CI, a setup check and shared run configurations. |
| 8. Docs | | | The README leads with the tested results; screenshots added. |

## What I learned about working this way

Short, specific requests worked best, along with "check this before you build on it" and sample sizes on every table. The agent did the long, exhaustive checks. I made the judgment calls, and the most important one was keeping the negative results visible instead of hiding them.
