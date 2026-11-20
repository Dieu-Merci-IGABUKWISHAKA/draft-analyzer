# Insights: data-backed counter-strategies

The **Insights** tab writes plain-language counter-strategies for the chosen opponent, then runs the same analysis on our own team. The logic is in `src/draft_analyzer/insights.py`.

The insights are built on **records and draft habits, not on the win model**: the model adds no reliable accuracy beyond team strength ([MODEL.md](MODEL.md)).

## Card types

Every rate on a card comes with a **Wilson 90% interval** and its game count. A card appears only if each group it compares has **at least 8 games**. Champions count as part of a team's pool from 5 picks, and a phase tell needs 6 picks.

| Card | Compares | Example advice |
|---|---|---|
| Ban impact | Their record when a pool champion is **open** vs **banned**. Games where it was already locked by Fearless are left out. | "Ban Yunara" |
| Fearless carry-over | Their record in games 2–5 while a top-5 champion is **still available** vs **already used** earlier in the series. Also for 2+ of their top 5 used up. | "Get Ezreal used up early in the series" |
| Comfort dependency | Their record with **2+ of their top 5 champions** drafted vs 0–1, plus the share of their picks in wins and in losses that are top-5 champions | "Deny their top 5" |
| Side / first pick | Their record on **blue vs red**, and **with vs without first pick**, plus how often each happens | "Put them on red" |
| Phase tell | Champions they take in the **first pick phase** in ≥ 85% of picks, or **hold for phase 2** (counter-pick) in ≥ 70% of picks. The interval must clear the 60% that a typical champion has. | "LYON take Nocturne early (7 of 7)" |

The wording is written separately for the opponent ("Ban X") and for self-scouting ("Opponents can ban X against us").

**Predictability:**
- **What's measured:** how spread out a team's **first picks** and **first bans** are, as entropy (Miller–Madow corrected, capped at the number of distinct champions seen), as the effective number of choices (2^entropy), and as the share covered by the top 3 champions.
- **Compared with:** every team with at least 15 games in the window.

## Which patterns are real? We checked

Many cards are tested per team, so some gaps will be luck. For every team, each pattern found in games **before** a split date was measured again on games **after** it. Two split dates were used, July 2025 and April 2026, and both were pooled. Later groups need only 3 games each.

The main table uses the cards that would be shown, those with **confidence ≥ 80%**. The last column covers every card of the kind.

| Kind of pattern | Confident checks | Same direction later | Gap before → after | Kept | All cards: same direction |
|---|---|---|---|---|---|
| Phase tell | 218 | **82%** | 40 → 23 pts (90% CI 20–25) | **57%** | 82% of 218 |
| Comfort dependency | 11 | **82%** | 27 → 13 pts (90% CI 5–21) | **48%** | 65% of 46 |
| Ban impact | 24 | **75%** | 26 → 8 pts (90% CI 1–15) | **30%** | 52% of 142 |
| First pick | 14 | 50% | 24 → 0 | 1% | 39% of 52 |
| Side | 16 | 44% | 23 → −2 | 0% | 46% of 52 |
| Fearless carry-over | 5 | 20% | 30 → −7 | 0% | 46% of 28 |

**What this means:**
- **Draft habits are real.** Phase tells point the same way in later games and keep more than half of their gap.
- **Comfort dependency holds up partly, and so do confident ban-impact gaps.** About a third to a half of the gap remains, on few checks: 11 and 24.
- **Most ban-impact gaps are chance.** Across all of them, only 52% point the same way later, but the confident ones mostly do.
- **Side, first pick and Fearless carry-over gaps don't hold up at all,** even when confident.

Because of this, each kind's weight in the ranking is measured on its confident cards: those are the ones a coach would see.

This matches the earlier findings: what a team *does* is predictable (their habits predict their next pick, see [SIMULATOR.md](SIMULATOR.md)), but single win-rate splits are noise at these sample sizes.

## Ranking: "How to beat <opponent>"

```
score = (gap / typical gap of its kind) × confidence × kept
```

- **Typical gap:** dividing by the typical gap of its kind puts win-rate gaps and pick-phase gaps on one scale.
- **Confidence:** 1 − p-value, from a two-proportion z-test (a one-proportion test against 60% for tells).
- **Kept:** the share of that kind's gap that held up in later games, from the table above.

**The bar:** a card is listed under "How to beat" only if the card has confidence ≥ 80% and its kind's confident cards kept at least 25% of their gap. Side, first-pick and Fearless carry-over cards therefore never qualify. I set that bar after seeing the replication table, and it is stated here and in the code. The top 5 cards that pass are shown. Every other pattern is still listed under "All patterns", marked as not holding up.

**In practice:** most "How to beat" cards are **phase tells**, with a confident ban-impact card now and then (for example "Don't spend a ban on Neeko" for FlyQuest). A comfort-dependency card appears when it is confident enough, which is rare at 60–80 games per team. That is what the data supports, not a design choice.

## The tab

1. **How to beat <opponent>:** up to 5 cards. Each shows the advice, the evidence with its 90% interval, the confidence, how much of that kind of pattern held up later, and the sample size.
2. **<Opponent>'s draft habits:** predictability of their first pick and first ban against the league.
3. **All patterns:** every card, including the ones that didn't hold up.
4. **Are these patterns real?** Two tables:
   - the walk-forward check of "likely picks" (top-5 hit rate 44% for picks and 38% for bans, against 37% and 26% for the league meta alone);
   - the replication table above.
5. **Self-scouting: Cloud9 Kia:** the same cards from an opponent's point of view, plus how predictable our first picks and bans are, as effective choices and against the league.

The window is the team's last N patches, 12 by default, because the cards need more games than the other tabs.

## Limitations

- **Associations, not causes.** A champion is banned more often by stronger opponents, a team plays blue more when it has the choice, and so on.
- **Thin samples:** 60–80 games per team in 12 patches. Most comparisons have wide intervals, which is why so few win-rate cards pass.
- **Few checks for some kinds:** the comfort-dependency (11) and ban-impact (24) weights rest on few confident checks. Fearless carry-over has only 5 confident checks, too few to rule out a small real effect.
- **Phase tells rest on small counts** (6–13 picks). Their kind replicates well, but any single tell can still be wrong.
- **The 25% / 80% bar was set after looking at the results** (see above).

`python src/draft_analyzer/insights.py` rebuilds the replication table (about 30 s; it caches aggregate numbers only in `data_cache/`). Tests: `tests/test_insights.py`.
