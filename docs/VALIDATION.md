# Validation: does it work?

Every number in this file comes from **games later than the ones used to build, fit or tune** what is being tested (AGENTS.md rule 5). Nothing here is a random split. Intervals are 90% or 95%, as labelled, and every result shows its sample size. The details are in the linked docs; this page collects them.

## At a glance

| What | How it was tested | Result | Verdict |
|---|---|---|---|
| **Ban plan** (top-5 bans) | Each 2026 game, computed from earlier games only. Weights tuned on 2025. | They picked **33.8%** of the champions we listed and left open, vs **19.8%** for their own top 5 and **26.9%** for the league meta | **Works.** +14.0 pts vs naive, +6.9 pts vs meta |
| **Likely picks** (simulator) | Every pick and ban of 897 games, each guessed from earlier games | Actual pick in our top 5 **44%** of the time (meta alone 37%); bans **38%** (meta 26%) | **Works** |
| **Draft tells** (Insights) | Patterns re-measured on later games | **82%** point the same way later, keeping 57% of the gap | **Works** |
| **Comfort dependency** and **confident ban-impact** gaps (Insights) | Same | 82% and 75% same direction, keeping 48% and 30% | **Partly** (few checks) |
| **Draft win model**: does the draft predict the winner? | Walk-forward by month, 897 games | Draft adds **no reliable accuracy** beyond team strength (log loss −0.004, CI −0.010 to +0.003) | **No.** Used only to show "what a choice shifts" |
| **Side, first pick, Fearless carry-over** gaps (Insights) | Re-measured on later games | 20–50% same direction, 0% kept | **No.** Not listed |
| **Early-game habits** (macro) | Re-measured on later games | Only first-tower rate and the jungler–first-grub link hold up | **Mostly no**. Two lines listed |

**The pattern:** what a team *does* in the draft (what it picks, bans, and when) is predictable from its history. *Which* draft wins is not, beyond team strength.

---

## 1. Ban plan backtest

`src/draft_analyzer/backtest.py`. Run it with `python src/draft_analyzer/backtest.py` (`--rebuild` rebuilds the cases, about 9 min).

**Setup.**
- **Cases:** for every game and both sides, the team we scout is "them" and the other team is "us".
- **The ban plan:** `recommend_bans`, computed only from games **before that game's date**, with the dashboard defaults (each team's last 6 patches, half-life 3).
- **Fearless:** champions already used earlier in the series are excluded from every list.
- **Burn-in:** a case needs at least 10 earlier games for the scouted team.
- **Sample:** 2025 (training) is 2,179 cases from 1,113 games. 2026 (test) is 836 cases from 439 games.

**The three lists compared** (top 5 each):
- the **Ban plan**;
- **naive:** their 5 most-picked champions in all earlier games;
- **league meta:** the 5 most-picked champions league-wide in the last 6 patches.

**The metric:** among the champions in a list that **we did not ban** and that were **still available to them** (not banned by either team, not picked by us, not Fearless-locked), how often they picked one.

**Tuning:**
- **Grid and objective:** 540 weight settings, scored on **2025 games only**, by that pick rate.
- **Fixed in advance:** the tuned weights would replace the hand-set ones if they beat them on 2025. They did: 37.0% vs 31.9%.
- **A grid adjustment, also on 2025 only:** the first grid had its best point on the edge. It was widened, and `prior_games` was fixed at 10 because 5, 10 and 20 differed by less than 0.2 pts on 2025.
- **The new defaults:** win_rate 3, pick 2, banned_by_others 1, our_pick_penalty 0, min_picks 2. Before: pick 0, banned_by_others 0.25, our_pick_penalty 0.5, min_picks 4.

### 2026 results (held out)

| List | Listed & available | They picked | Pick rate (90% CI) | Games where they picked ≥ 1 | Real team banned it too |
|---|---|---|---|---|---|
| **Ban plan (tuned on 2025, now the default)** | 2,015 | 681 | **33.8%** (32.1–35.6) | **63.8%** (513 of 804) | 26.3% |
| Ban plan (previous hand-set weights) | 2,299 | 609 | 26.5% (25.0–28.0) | 56.4% (457 of 811) | 20.3% |
| League meta: top 5 picks | 2,476 | 667 | 26.9% (25.5–28.4) | 60.0% (500 of 833) | 12.5% |
| Naive: their 5 most-picked ever | 2,791 | 553 | 19.8% (18.6–21.1) | 54.1% (451 of 833) | 13.6% |

**Paired differences in pick rate** (2026, bootstrap over cases, 90% CI):
- **Tuned Ban plan vs naive:** **+14.0 pts** (12.3 to 15.7).
- **Tuned Ban plan vs league meta:** **+6.9 pts** (4.9 to 8.9).
- **Tuned vs previous weights:** **+7.3 pts** (5.8 to 8.8).
- **Previous weights vs meta:** −0.4 pts (−2.3 to +1.4). The old hand-set plan was no better than the meta.

### Are the listed champions also dangerous?

This checks their win rate in games where they got a listed champion that we left open:

| List | 2025: win rate when they got one / overall | 2026: win rate when they got one (90% CI) / overall | Excess wins vs their pre-game rate, 2025 → 2026 |
|---|---|---|---|
| Ban plan (tuned) | 50.2% / 50.2% | 53.2% (50–57), 273–240 / 50.5% | −0.019 → +0.020 |
| Ban plan (previous) | 51.4% / 50.2% | 54.3% (50–58), 248–209 / 50.5% | −0.014 → +0.021 |
| Naive | 50.0% / 50.2% | 51.2% (47–55) / 50.5% | −0.017 → +0.001 |
| Meta | 49.6% / 50.2% | 49.2% (46–53) / 50.5% | −0.018 → −0.015 |

**No consistent evidence.** In 2025, teams did no better on the champions we would have banned. In 2026 they did a little better, by 2–3 points, but the intervals reach 50%. This matches the Insights replication: per-champion win-rate gaps are mostly noise.

**So the Ban plan's proven value is that it removes champions the opponent actually plays.** It is not proven that those champions win them more games.

## 2. Draft win-probability model

`src/draft_analyzer/model.py`; details in [MODEL.md](MODEL.md). Walk-forward by month: each month is trained on all earlier games. That gives 13 reported months, July 2025 to October 2026, **897 games**. Intervals are series bootstrap, 95%.

| Model | Accuracy | Log loss (lower = better) | Brier |
|---|---|---|---|
| Elo only (team strength) | 64.1% (60.8–67.7) | **0.6389** (0.613–0.663) | **0.2235** |
| Elo + side | 65.3% (62.0–68.7) | 0.6411 | 0.2243 |
| Full model (Elo + side + draft) | 65.2% (62.0–68.5) | 0.6449 | 0.2253 |

**What the draft adds over Elo + side:**
- **Accuracy:** −0.1 pts (−2.0 to +1.7).
- **Log loss:** −0.004 (−0.010 to +0.003).
- **Verdict:** no reliable gain, in any league.

A whole draft moves the estimate by a median of **3.0 points**. That's why the simulator shows "what each choice shifts" and never a forecast.

## 3. Likely picks and bans (Draft simulator, Insights)

`simulator.likely_accuracy`; details in [SIMULATOR.md](SIMULATOR.md). This covers every pick and ban of **897 games** from July 2025 on. Each one was guessed from games on earlier dates, with Fearless locks applied:

| | Checked | Top guess right | Actual in our top 5 | League meta alone: top 5 |
|---|---|---|---|---|
| Picks | 8,970 | 13% | **44%** | 37% |
| Bans | 8,964 | 11% | **38%** | 26% |

## 4. Insight patterns: do they hold up later?

`insights.replication`; details in [INSIGHTS.md](INSIGHTS.md). For each team, patterns were found in games before July 2025 or April 2026, then measured again on later games. Confident patterns only (≥ 80%).

**A kind of pattern holds up** if it kept at least 25% of its gap **and** the later gap's 90% interval is above zero.

| Kind | Checks | Same direction later | Gap before → after | Kept | Holds up |
|---|---|---|---|---|---|
| Phase tell (picked early / held back) | 218 | **82%** | 40 → 23 pts | **57%** | Yes |
| Comfort dependency | 11 | **82%** | 27 → 13 pts | **48%** | Yes |
| Ban impact | 24 | **75%** | 26 → 8 pts | **30%** | Yes (all ban gaps, including weak ones: 52%) |
| First pick | 14 | 50% | 24 → 0 | 1% | No |
| Side | 16 | 44% | 23 → −2 | 0% | No |
| Fearless carry-over | 5 | 20% | 30 → −7 | 0% | No |

## 5. Early-game patterns: do they hold up later?

`macro.replication`; details in [MACRO.md](MACRO.md). Same method as section 4.

| Pattern | Checks | Same direction | Kept | Holds up |
|---|---|---|---|---|
| First-tower rate | 31 | **74%** | **53%** | Yes (mostly team strength) |
| Jungler ↔ first void grub | 19 | **74%** | **27%** | Yes, barely (later-gap CI +1 to +18 pts) |
| First-grub rate, dragon timing, herald timing | 7–17 | 57–73% | 25–61% | No (later-gap CI includes 0) |
| First blood, dragon and herald rates; other jungler links; other timings | 13–33 | 39–85% | 0–22% | No |

## 6. Other checks

- **Data source:** the final snapshot and the event stream agree on the draft in 1,734 of 1,739 games. The snapshot is used because it resolves undo events and missing bans.
- **Inferred roles:** 100% correct on held-out events. Each labelled event was predicted from the others. This is not a time split; the roles are a labelling step, not a forecast.
- **Fearless rule:** detected from the data. There are 0 repeated champions in 568 multi-game series, across 20 events.
- **Leakage tests** (`tests/`, synthetic data, `python -m pytest`):
  - **No-peeking tests:** model features, walk-forward months, the likely-pick tendencies, insight replication, macro extraction and Ban plan cases. Each checks that results for game N are unchanged when later games are removed or their results flipped.
  - **The tests were themselves checked:** planted leaks are caught. One of the walk-forward tests was added after the first version missed a leak.

## How to reproduce

```bash
python src/draft_analyzer/drafts.py --rebuild     # drafts table + early-game events (~40 s)
python src/draft_analyzer/model.py                # win model, walk-forward (~30 s)
python src/draft_analyzer/simulator.py            # likely-pick accuracy (~1 min)
python src/draft_analyzer/insights.py             # insight replication (~30 s)
python src/draft_analyzer/macro.py                # early-game replication (seconds)
python src/draft_analyzer/backtest.py --rebuild   # Ban plan backtest (~10 min)
python -m pytest                                  # all tests (~2-3 min)
```

The scripts cache only in `data_cache/`, which is gitignored. Only the aggregate numbers above are committed.
