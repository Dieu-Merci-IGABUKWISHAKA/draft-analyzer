# Early game: macro context from the event feeds

A short **Early game** section at the bottom of the **Opponent scouting** tab adds supporting context from the in-game event feeds. The logic is in `src/draft_analyzer/macro.py`. As in [INSIGHTS.md](INSIGHTS.md), **only patterns that held up in later games are listed**.

## What is extracted, per game

| First… | Event in the feed | Coverage |
|---|---|---|
| Blood | `player-killed-player` (executions and multikill events left out) | 100% of 1,734 games |
| Dragon | `player-completed-slayDragon`, or any drake-type event such as `slayMountainDrake` | 100% |
| Void grub | `player-completed-slayVoidGrub` (the team that killed the first grub) | 99.9% |
| Rift herald | `player-completed-slayRiftHerald` | 99.5% |
| Tower | `player-destroyed-tower` or `team-destroyed-tower` (minions) | 100% |

- **Team:** the acting player's team, or the acting team. A tower is credited to the side opposite its owner (its id starts with `blue-` or `red-`), because in 8 of 20,647 tower events the credited team was on the tower's own side.
- **Game time:** every event carries a full snapshot, and the game's `clock.currentSeconds` is read from it.
- **Dragon events:** feeds from the first events of 2025 (LCK Cup, LEC Winter and LTA Split 1, 263 games) have no generic `slayDragon` event, only the drake-type ones. Both are collected, and the earliest counts.
- **Undo events (`!`):** they are collected and applied. There are none for these event types.
- **Sanity check:** median times look right: first blood about 5:00, first dragon about 7:00, first grub about 8:30, first tower and herald about 16:00.

**Cost:** the events are collected in the **same pass as the drafts table** (`drafts.scan_series`), and the full rebuild still takes about 40 s. They are cached in `data_cache/objective_events.parquet`, which is gitignored and contains row-level data.

## Patterns

These cover each team's last 12 patches:

- **Objective rate:** how often they take each objective first, compared with 50%. Confidence comes from a one-proportion test.
- **Objective timing:** their median time when they take it first, compared with the league median over the same dates. Confidence comes from a sign test.
- **Jungler link:** their rate with one jungler champion against their rate with their other junglers. It needs at least 5 games each way, and confidence comes from a two-proportion test.

## Which patterns held up? We checked

The check is the same as in INSIGHTS.md:
- For every team, each pattern found in games before July 2025 or April 2026 was measured again on later games, where later groups need 3 games.
- Only patterns with confidence ≥ 80% are counted.
- **A kind holds up** if it kept at least 25% of its gap **and** the 90% interval of the later gap is above zero. The second condition was added in this phase. All three Phase 3 kinds that passed also meet it, so the Insights tab doesn't change.

| Pattern | Objective | Confident checks | Same direction later | Gap before → after | Kept | Held up |
|---|---|---|---|---|---|---|
| Objective rate | First tower | 31 | **74%** | 16 → 8 pts (90% CI 5 to 12) | **53%** | **Yes** |
| Objective rate | First void grub | 11 | 73% | 13 → 8 pts (CI −1 to 18) | 61% | No (CI includes 0) |
| Objective rate | First herald | 14 | 57% | 15 → 3 pts | 21% | No |
| Objective rate | First dragon | 18 | 39% | 12 → 1 pt | 7% | No |
| Objective rate | First blood | 15 | 53% | 13 → 0 | 0% | No |
| Jungler link | First void grub | 19 | **74%** | 35 → 9 pts (CI 1 to 18) | **27%** | **Yes (barely)** |
| Jungler link | First herald | 26 | 58% | 35 → 4 pts | 13% | No |
| Jungler link | First blood | 26 | 54% | 33 → 4 pts | 12% | No |
| Jungler link | First dragon, first tower | 19 / 33 | 53% / 52% | ≈ 0 | 0% | No |
| Timing | First dragon | 17 | 65% | 70 → 22 s (CI −10 to 55) | 32% | No (CI includes 0) |
| Timing | First herald | 7 | 57% | 21 → 5 s | 25% | No (CI includes 0) |
| Timing | First tower | 13 | 85% | 55 → 12 s (CI −1 to 25) | 22% | No |
| Timing | First blood | 14 | 57% | 77 → 13 s | 16% | No |
| Timing | First void grub | 15 | 47% | 42 → 1 s | 3% | No |

**What this means:**
- **Most early-game habits are not stable enough to plan around.** That includes most "with jungler X they take Y" lines, which would have looked convincing on 5–10 games.
- **Two patterns hold up:**
  - **First-tower rate**, which mostly reflects team strength.
  - **The jungler–first-grub link**, only just: the interval's lower end is +1 point.
- **The near misses:** first-grub rate, dragon timing and herald timing kept a quarter or more of their gap, but on few checks, and their later gap could still be zero.

## What the tab shows

- **The list:** a line appears only if its kind held up **and** the line itself is confident (≥ 80%) for this opponent. For the teams checked:
  - **LYON:** nothing passes.
  - **FlyQuest:** "With Ambessa, FlyQuest take the first void grub 20% of the time (1 of 5 games) vs 57% with other junglers (60 games)."
  - **Cloud9 Kia:** first tower 58% (45 of 78) passes.
- **Labels:** every line shows its sample size and its 90% range.
- **An expander** shows every early-game number (rates, their median times, league medians) together with the replication table above, so nothing is hidden.

## Limitations

- **Small samples:** about 70 games per team in 12 patches, and 5–10 games per jungler champion.
- **What "first" means:** first grub is the team that killed the first grub, not the one that took the most of the first spawn. First blood ignores executions.
- **Rates reflect team strength:** a strong team takes more first objectives everywhere.
- **Patch changes move spawn timers:** grubs come about 45 s later in 2026. That's why timings are compared with the league over the same dates.
- **Not modelled:** objective trades (giving dragon for grubs) and lane assignments.

**Commands:**
- `python src/draft_analyzer/macro.py` rebuilds the replication table (seconds; it caches aggregate numbers only).
- `python src/draft_analyzer/drafts.py --rebuild` refreshes the drafts table and the events.
- Tests are in `tests/test_macro.py`. The extraction is tested on a tiny synthetic feed.
