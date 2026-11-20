# AI workflow: one afternoon, one human, one coding agent

The project was built on 20 November 2026, between about 15:00 and 17:00 (see the git log). The work was split simply. I decided which questions mattered to a coach, set the constraints and checked every result. Claude Code, running in PyCharm's terminal, read the data docs, wrote and ran the code, and reported back with numbers I could verify.

## Where the AI did the heavy lifting

- **License-safe setup.** The data was cloned sparsely next to the project, and the `.gitignore` blocks every data format. The agent ran the data repo's own check before every commit, and scanned the full history before the first public push.
- **Choosing the data source with evidence.** Instead of sampling 2–3 series, the agent compared the event stream with the final snapshot across all 745 series. They matched in 1,734 of 1,739 games. The stream missed bans in the other 5 and carried 461 undo events, so the snapshot won.
- **Finding the 2026 first-selection change.** The first draft-slot mapping assumed blue side picks first, and over 200 games didn't fit. The agent traced them all to 2026, where the first-pick team is often red, and re-keyed the slots on "the team that picks first".
- **Detecting Fearless from the data.** Instead of hard-coding which leagues use Fearless Draft, it checked for repeated champions within each series. There were none in 568 multi-game series across all 20 events.

## Where I had to steer

- **Roles.** Drafts have no role field. I told the agent to tell me before inventing one, and it stopped to ask. Its first idea (roster order) turned out to be role order only in early-2025 events: Cloud9's 2026 list started with their support. So it switched to learning roles from the labelled events and inferring the rest. It reported a held-out accuracy of 100% before we used them.
- **The ban score.** Version 1 ranked Nocturne #2 against LYON despite LYON's 44% win rate on it, because "other teams ban it" weighed too much. I asked for that weight to drop and for losing records to count against a ban. Version 2 then put a 3–0 LeBlanc and a 4–0 Varus on top. I asked for stronger smoothing and a minimum-picks rule, until Ryze (6–1 in 7 games) outranked Varus (4–0).
- **Honesty checks.** Before writing these docs, the agent re-checked its claims. It flagged an unused library (duckdb) and one of its own earlier counts that was wrong (663 instead of 568 series).

The pattern that worked: short, specific requests, "check this before you build on it", and sample sizes on every table. The agent did the exhaustive checks; I made the judgment calls.
