# Draft win-probability model

`src/draft_analyzer/model.py` estimates **the chance that blue side wins a game**, from how strong the two teams are, the side, and the draft so far. It works mid-draft: give it the picks and bans made so far and it updates.

**Short version, tested on 897 games it never saw (July 2025 to October 2026):**
- **Team strength (Elo) alone** predicts 64.1% of games correctly.
- **Adding the draft does not make it reliably better.** The full model gets 65.2%, and its probabilities are slightly *worse* by log loss (0.645 vs 0.639). Every difference sits inside the confidence intervals.
- **So the draft gain is not statistically real.** The draft features do carry signal of their own: teams with more experience on their picks win more. But that signal is mostly the same information Elo already has.

**What this means for the simulator:** present each pick as **"how much this choice shifts the odds, according to past games"**, with the sample size behind it. Do not present it as a confident forecast of who will win.

Run it: `python src/draft_analyzer/model.py` (about 25 s). It prints the full report and saves the model to `data_cache/draft_model.joblib` (gitignored). Tests: `python -m pytest`.

## What changed from the first version

| | Version 1 (Phase 1) | Now (Phase 1b) |
|---|---|---|
| Target | First-pick team wins | **Blue side wins** |
| Side and first pick | One mixed feature | **Separate.** Side is the model's intercept; `blue_first_pick` is its own feature. Since 2026 the first pick can be on red side. |
| Draft features | Champion win rate, presence, team comfort, bans | Added **player comfort**, **lane matchups** and **first-phase priority** |
| Testing | One split: 275 test games | **Walk-forward by month: 897 test games**, with confidence intervals |
| Calibration | None | **Platt scaling**, fitted only on earlier months |

Version 1 showed a small gain from the draft on its single test period (log loss +0.011). **That gain does not hold up on the larger walk-forward test.** That is why one split wasn't enough.

## Features

**No leakage.** Every feature for a game uses **only games played on earlier dates**. That leaves out the game itself, other games the same day, and anything later. The tests check this (see [Tests](#tests)).

Each draft feature is **blue minus red**, summed over the picks or bans made so far.

| Feature | Plain meaning | Built from |
|---|---|---|
| `elo_diff` | Which team is stronger right now | An Elo rating per team, updated after every game. K = 40 was chosen on the 570 games before July 2025. |
| Side (the intercept) | Does blue side win more? | Fitted constant |
| `blue_first_pick` | Does blue make the first pick? | Draft slot 7. Always yes before 2026, often no in 2026. |
| `d_champ_wr` | Is the champion strong in the current meta? | Win rate across all teams in the last 3 patches, pulled toward 50% by 20 imaginary games |
| `d_presence` | Is it a contested champion? | Share of games in those 3 patches where it was picked or banned |
| `d_priority` | Do teams take it early? | Share of its picks made in the first pick phase, smoothed, minus the 60% expected by chance |
| `d_team_games`, `d_team_edge` | Does the team know this champion in this role? | The team's games on it in that role over the last 365 days (log scale), and its smoothed win rate on it minus its own average |
| `d_player_games`, `d_player_edge` | Does **the player** know this champion? | The same for the player who picked it, compared with that player's own average |
| `matchup` | Who wins this lane matchup? | For each role where both sides have picked: the smoothed past win rate of blue's champion against red's in that role (365 days, 10 imaginary games at 50%). It counts as 50% (no effect) if they met fewer than 3 times. |
| `d_denied` | Did bans hit the opponent's pool? | For each ban: log(1 + games the opponent played on that champion) |
| `d_n_picks`, `d_n_bans` | How far the draft has got | Counts. They matter for partial drafts. |

No hand-made champion tags (such as "early game" or "scaling") are used. Roles come from the inferred roles (see README). Players come from the rosters.

**Partial drafts.** Features are sums, so a pick not made yet adds nothing. Each game appears in training once after each of 0 to 20 draft actions, with weight 1/21, so each game counts once in total.

### Are the features working? A sanity check

Each feature is checked on its own, on the 1,450 finished drafts from April 2025 on. AUC measures how well one number alone separates wins from losses: 0.5 means no signal.

| Feature | Non-zero in | AUC alone | Correlation with Elo |
|---|---|---|---|
| Elo | 99% | 0.70 | 1.00 |
| Player games on champion | 98% | **0.60** | 0.36 |
| Team games on champion | 99% | 0.57 | 0.27 |
| Team results on champion | 99% | 0.55 | 0.18 |
| Player results on champion | 100% | 0.54 | 0.18 |
| Champion meta win rate | 100% | 0.53 | 0.05 |
| Lane matchups | 99% | 0.53 | 0.06 |
| Priority | 100% | 0.52 | 0.01 |
| Presence | 100% | 0.51 | 0.07 |
| Bans hitting opponent's pool | 99% | 0.40 (reversed) | −0.34 |

The features are filled in and point the expected way, so this is not a bug. The strongest draft signals (comfort) are partly team strength in disguise: good teams have experienced players. The bans feature is reversed because underdogs ban the favourite's best champions. This table describes all games, including test months, and was not used to choose anything.

## Validation: walk-forward by month

- **For each calendar month:** train on **every earlier game** and test on that month.
- **Months tested:** 16 in all. The first 3 (April–June 2025, 553 games) only warm up the settings. The **13 reported months, July 2025 to October 2026, hold 897 games**. October to December 2025 has no games.
- **Choices made from earlier months only:** the regularisation strength C (picked from 0.003–1 by log loss), and the Platt calibration of each model's probabilities. Two tests prove no month sees its own or later results.
- **Elo K and the feature settings** were fixed before the reported months: K was tuned on games before July 2025, and the rest were set in advance.
- **Confidence intervals** are 95%, by bootstrap over **series**, not games, because games in the same series aren't independent.

**Three models:**
- **Elo only:** no intercept, so no side information.
- **Elo + side:** adds side and first pick.
- **Full:** Elo + side + all draft features.

### Results (all 897 test games, full drafts)

| Model | Accuracy | Log loss | Brier |
|---|---|---|---|
| Coin flip (reference) | 50% | 0.6931 | 0.2500 |
| **Elo only** | 64.1% (60.8–67.7) | **0.6389** (0.613–0.663) | **0.2235** (0.212–0.234) |
| Elo + side | 65.3% (62.0–68.7) | 0.6411 (0.615–0.666) | 0.2243 (0.213–0.235) |
| **Full model** | 65.2% (62.0–68.5) | 0.6449 (0.616–0.674) | 0.2253 (0.213–0.238) |
| Full model, without calibration | 64.7% (61.4–68.1) | 0.6413 (0.614–0.668) | 0.2242 (0.213–0.236) |

How to read the metrics:
- **Accuracy:** share of games where the side given over 50% won.
- **Log loss** and **Brier score:** how good the probabilities are. Lower is better.

### Is the draft gain real?

The table shows the gain of the second model over the first. A positive number means better. "Chance of no gain" is the share of bootstrap samples where the second model wasn't better.

| Comparison | Accuracy | Log loss | Chance of no gain (log loss) |
|---|---|---|---|
| Full vs Elo + side (what the draft adds) | −0.1 pts (−2.0 to +1.7) | −0.004 (−0.010 to +0.003) | 86% |
| Full vs Elo only | +1.1 pts (−1.0 to +3.2) | −0.006 (−0.014 to +0.001) | 94% |
| Elo + side vs Elo only (what side adds) | +1.2 pts (−0.3 to +2.9) | −0.002 (−0.006 to +0.002) | 88% |

**Plainly: no.** Adding the draft does not improve the predictions on games the model hasn't seen. If anything, the probabilities get slightly worse (log loss), though that difference is also within the noise. Side and first pick add nothing reliable either. This fits what is known about pro play: both teams draft competently, so the draft rarely decides a game by itself. Execution and team strength do.

### By league

| League | Games | Elo only: accuracy / log loss | Elo + side | Full model |
|---|---|---|---|---|
| LCK | 206 | 67.5% / 0.615 | 68.0% / 0.618 | 67.0% / 0.626 |
| LEC | 324 | 67.3% / 0.631 | 67.6% / 0.635 | 67.0% / 0.635 |
| LCS | 210 | 60.5% / 0.672 | 63.8% / 0.675 | 63.8% / 0.677 |
| LTA | 157 | 58.0% / 0.642 | 59.2% / 0.639 | 61.1% / 0.648 |

Per-league accuracy intervals are about ±6 to ±9 points (the script prints them), so no league shows a reliable draft gain. Full vs Elo + side log-loss gains are LCK −0.008, LEC −0.000, LCS −0.001 and LTA −0.009, and every interval includes zero. The LCS and LTA are harder to predict than the LCK and LEC, and their sample sizes are smaller.

### Calibration

Platt calibration was fitted on earlier months only. Test games are split into five equal groups by the predicted chance that blue wins:

| Predicted | Games | Full model: predicted | Actual | Without calibration: predicted | Actual |
|---|---|---|---|---|---|
| lowest fifth | 180 | 23.5% | 27.8% | 25.5% | 27.2% |
| 2nd | 179 | 39.1% | 38.0% | 41.2% | 40.2% |
| middle | 179 | 51.4% | 62.0% | 52.9% | 59.8% |
| 4th | 179 | 63.0% | 60.3% | 63.9% | 60.9% |
| highest fifth | 180 | 79.0% | 73.3% | 79.0% | 73.3% |

- **Calibration did not help here:** log loss went from 0.641 without it to 0.645 with it. The plain logistic regression was already close to calibrated, and a calibration learned on earlier months doesn't transfer perfectly to later ones.
- **The final model's calibration is nearly a no-op:** slope 0.985, shift −0.024, fitted on all out-of-sample predictions. It does no harm, so it is kept.
- **Strong favourites win somewhat less often than predicted:** 73% against 79% in the top fifth. That gap shows with or without calibration.

### By draft stage (pooled test months)

| Draft actions done | Accuracy | Log loss |
|---|---|---|
| 0 (no draft yet) | 64.7% | 0.6418 |
| 6 (first bans) | 65.1% | 0.6418 |
| 12 (3 picks each) | 65.2% | 0.6422 |
| 16 (all bans) | 65.0% | 0.6423 |
| 20 (full draft) | 65.2% | 0.6449 |

The draft barely changes the quality of the prediction. From no draft to the full draft, a test game's estimate moves by a **median of 3.0 percentage points**: 90% of games move less than 7.6 points, and the largest move is 16.8.

### Sides

| In the 897 test games | |
|---|---|
| Blue side win rate | 52.3% |
| Games where blue has first pick | 73% |
| Blue win rate when blue has first pick | 51.8% |
| Blue win rate when red has first pick (243 games) | 53.5% |

In this data, first pick is not an advantage, and blue side is only a small one.

## Which features carry weight

**Permutation importance:** how much worse the test log loss gets when one feature is shuffled. Each test month uses its own model. Higher means the model leans on it more, and a negative value means the feature hurt.

| Feature | Log-loss rise |
|---|---|
| Elo | **+0.111** |
| Team results on champion | +0.004 |
| Player games on champion | +0.004 |
| **All draft features together** | **+0.003** |
| Player results on champion | +0.002 |
| Team games on champion | +0.001 |
| Presence, bans, matchups | ≈ 0 |
| Priority, blue first pick | ≤ 0 |
| Champion meta win rate | −0.001 |

**Coefficients of the final model** (fitted on all games, C = 0.03, standardised; associations, not causes): Elo +0.62. Then team results on champion +0.10, player games +0.09, bans −0.08, player results −0.05, presence −0.05, priority +0.05, team games −0.04, matchups +0.03, blue first pick −0.03, meta win rate +0.02. The intercept is +0.12, a small blue-side edge.

Elo carries almost everything. Within the draft, **comfort (team and player experience and results on their picks)** is the only group with weight. Meta strength, priority and lane matchups add nothing measurable once team strength is known.

## Live use: `predict_draft`

```python
from draft_analyzer.model import predict_draft

r = predict_draft("Cloud9 Kia", "LYON", side_a="blue",
                  picks_a=["Ryze"], picks_b=["Cassiopeia", "Ziggs"],
                  bans={"a": ["Varus"], "b": ["Azir"]})
r["p_team_a"]            # P(Cloud9 wins), given the draft so far (calibrated)
r["p_team_a_elo_only"]   # the same from team strength alone
r["push"]                # each feature's push on Cloud9's log-odds vs an average game
r["roles"]               # role used for each pick (guessed unless roles_a/roles_b given)
r["actions"]             # each pick/ban with the sample sizes behind every number
```

- **Inputs:**
  - Teams by name or GRID id; champions by name or id; 0–5 picks per side.
  - Bans are a flat list, or `{"a": [...], "b": [...]}` when you know who banned what.
  - `roles_a` / `roles_b` set the role of each pick. Otherwise the most likely distinct roles are guessed from the team's and all teams' history. The **player** for each role comes from the team's most recent lineup.
  - `date` sets the history cutoff: only earlier games are used. `patch` ends the meta window at that patch. `first_pick` is "a" or "b", and defaults to blue.
- **Sample sizes** are returned for every number, as AGENTS.md rule 4 requires:
  - champion picks and wins in the meta window, and the window size;
  - the team's and the player's games and wins on the champion;
  - lane-matchup games and wins against the opposing champion;
  - total history games.
- **Checks:** it raises a clear error for unknown teams or champions, more than 5 picks, a champion used twice, or invalid roles.

## Tests

`tests/test_model.py`. The tests use a made-up drafts table, so no licensed data is needed or stored:

- **Features use only earlier games.** Features for game N are identical when every later game is removed (three cut points), and when the results of all games on game N's date are flipped.
- **Walk-forward uses only earlier months.** Removing later months leaves each month's predictions, chosen C and calibration unchanged. Flipping a month's own results leaves that month's predictions, C and calibration unchanged. I checked both tests by planting leaks. A month that uses its own results is caught by the second test only, which is why both exist.
- **Partial drafts:**
  - Stage 0 has no draft signal; picks count up; a matchup appears only once both lane picks exist.
  - Valid probabilities for 0v0 to 5v5, and swapping the teams gives exactly 1 − p.
  - `date=D` ignores games on or after D, explicit roles override the guess, and bad drafts are rejected.
- **Real data:** the no-leakage test on the real drafts table, skipped when `data_cache/drafts.parquet` is missing.

## Limitations

- **The draft signal is not distinguishable from noise** on 897 games, once team strength is known. A much larger sample might find a small real effect. This one can't.
- **Comfort features are partly team strength:** experienced teams are better teams. Their pushes in `predict_draft` are associations, not proof that a pick causes a win.
- **Roles and players at draft time:** training uses each game's actual (inferred) roles and players. Live, roles are guessed and players come from the last lineup, so flex picks and substitutions can be wrong unless given explicitly.
- **Elo carries over between seasons** with no reset, although rosters change. It is only comparable within a league.
- **Some months are tiny**, for example July 2026 with 2 games. Results are pooled across months for that reason.
- **Fixed settings:** windows and smoothing were fixed in advance, not tuned. Trying many would mostly fit noise.
- **Who held first selection** in 2026, and whether they chose side or first pick, is not in the data.
