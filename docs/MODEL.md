# Draft win-probability model

`src/draft_analyzer/model.py` estimates **the chance that the team with first pick wins the game**, from how strong the two teams are and from the draft so far. It also works mid-draft: give it the picks and bans made so far and it updates, so it can drive a live "win chance" display while a coach enters a draft.

**Short version:** the model predicts 65% of 2026 games correctly on games it never saw. Almost all of that comes from team strength (Elo). The draft adds a little, but not reliably: the 95% confidence interval of the gain includes zero. That matches what is known about pro play: both teams draft well, so drafts rarely decide a game on their own.

Run it: `python src/draft_analyzer/model.py` (about 25 s). It prints the report below and saves the model to `data_cache/draft_model.joblib` (gitignored). Tests: `python -m pytest`.

## What it predicts

- **Unit:** one row per game, seen from the **first-pick team** (the team that makes pick 1, draft slot 7). The target is 1 if that team won.
- **Data:** all 1,734 games in the drafts table (LCK, LEC, LTA and LCS, January 2025 to October 2026).
- `predict_draft` turns this into "P(team A wins)" for any two teams, either side, and any partial draft.

## Features

**No leakage.** Every feature for a game uses **only games played on earlier dates**. That leaves out the game itself, other games on the same day, and anything later. Two tests check this (see [Tests](#tests)).

| Feature | Plain meaning | Built from |
|---|---|---|
| `elo_diff` | Which team is stronger right now | An Elo rating per team. Every win moves rating from the loser to the winner, more for an upset. The step size K = 40 was tuned on 2025 games only. |
| `fp_blue` | Is the first-pick team on blue side? | Before 2026 the first-pick team was always blue. In 2026 it is often red. |
| `d_champ_wr` | Are their champions winning right now? | Each picked champion's win rate across all teams in the last 3 patches, pulled toward 50% by 20 imaginary games, minus 50%. Summed over the team's picks. |
| `d_presence` | Did they get high-priority champions? | How often each picked champion is picked or banned in those 3 patches. |
| `d_comfort_games` | Has this team played these champions? | log(1 + the team's games on the champion over the last 365 days). It is **role-aware**: only games in the role the team usually plays that champion in count. |
| `d_comfort_edge` | Does this team win on these champions? | The team's win rate on the champion (pulled toward its overall win rate by 5 imaginary games) minus its overall win rate. |
| `d_denied` | Did their bans hit the opponent's pool? | For each of the team's bans: log(1 + games the opponent played on that champion). |
| `d_n_picks`, `d_n_bans` | How far the draft has got | Count of picks and bans each side has made. This matters for partial drafts. |

Each `d_` feature is **first-pick team minus second-pick team**. The draft features are **sums** over picks (not averages), so a pick that hasn't happened yet adds nothing. That is what lets one model score a draft at any point.

We used no hand-made champion tags (such as "early game" or "scaling"), because the data doesn't contain them and inventing them would be guesswork.

**Partial drafts in training.** Each game appears 21 times in training, once after each of 0, 1, …, 20 draft actions. Each copy has weight 1/21, so every game counts once in total. This keeps the model's probabilities honest mid-draft, not just at the end. On the inner validation split, the same model trained only on finished drafts did worse (log loss 0.715 vs 0.686).

## Validation

Everything uses time-based splits: fit on earlier games, test on later ones (AGENTS.md rule 5).

| Step | Games used |
|---|---|
| Tune Elo K | One-step-ahead predictions on 1,017 games from 2025, skipping the first 60 days while every rating is still 1500 |
| Choose the model (logistic regression C, or gradient boosting) | Fit on 2025 games, compare on 175 games from 2026-01-01 to 2026-03-31 |
| Final fit | All 1,459 games before 2026-04-01 |
| **Test (reported once)** | **275 games from 2026-04-04 to 2026-10-13**: LEC Spring 2026 (from April), LCS Split 2, Split 3 and Promotion 2026 |

The test games were not used for any fitting, tuning or model choice.

**Model choice:** logistic regression with strong regularisation (C = 0.01) had the best inner-validation log loss (0.686). Gradient boosting scored 0.702, and the weaker-regularised logistic regressions scored 0.708–0.713. All of these are close to a coin flip (0.693) on that small, hard period, the first months of a new season. So the choice rests on thin evidence. On the test set gradient boosting was clearly worse too (below).

### Test results

| Model | Games | Accuracy | Log loss | Brier | Calibration error |
|---|---|---|---|---|---|
| Coin flip (for reference) | 275 | – | 0.6931 | 0.2500 | – |
| Raw Elo formula | 275 | 64.4% | 0.6392 | 0.2224 | 0.051 |
| **Baseline: Elo only** (logistic regression) | 275 | 64.4% | 0.6423 | 0.2243 | 0.066 |
| Elo + side | 275 | 66.5% | 0.6367 | 0.2215 | 0.052 |
| **Main: Elo + side + draft** | 275 | 65.1% | **0.6312** | **0.2195** | 0.061 |
| Gradient boosting, same features (not chosen) | 275 | 60.7% | 0.6723 | 0.2378 | 0.089 |

How to read the metrics:
- **Accuracy:** share of games where the team given over 50% won.
- **Log loss** and **Brier score:** how good the probabilities are. Lower is better. A confident miss costs a lot in log loss.
- **Calibration error:** the average gap between predicted and actual win rate across the five bins below.

With 275 games, accuracy has a margin of error of about ±5.6 points (95%), so differences of 1–2 points between models are noise.

### How much the draft adds on top of Elo

Paired bootstrap over the 275 test games, 2,000 resamples:

| Comparison | Log-loss gain (95% CI) | Accuracy gain (95% CI) |
|---|---|---|
| Main vs Elo only | +0.011 (−0.002 to +0.025) | +0.7 pts (−2.5 to +3.6) |
| Main vs Elo + side | +0.006 (−0.007 to +0.019) | −1.5 pts (−5.1 to +1.8) |

**Honest reading:** the draft features improve the probabilities a little (lower log loss and Brier), but the gain is **not statistically clear**: both intervals include zero. They don't improve accuracy. As a live tool, going from no draft to the full draft moves a test game's estimate by a **median of 2.3 percentage points** (90% of games under 5.7, at most 8.8). The team-strength gap is what sets the number.

The regularised coefficients point the expected way. They are on a standardised scale, larger means more weight, and they are associations, not causes:

| Feature | Coefficient | Reading |
|---|---|---|
| Elo | +0.51 | Dominant |
| Team's own results on its picks | +0.06 | |
| Team experience on its picks | +0.03 | |
| Champion win rate this window | +0.03 | |
| Champion priority | +0.01 | |
| Side | +0.02 | |
| Bans that hit the opponent's pool | −0.05 | Probably because underdogs spend bans on the favourite's best champions, not because those bans hurt |

### Calibration

Test games split into five equal groups by the predicted chance that the first-pick team wins:

| Predicted range | Games | Elo only: predicted | Actual | Predicted range | Games | Main: predicted | Actual |
|---|---|---|---|---|---|---|---|
| 12–38% | 55 | 27.8% | 18.2% | 17–40% | 55 | 31.6% | 21.8% |
| 38–49% | 55 | 43.3% | 41.8% | 40–49% | 55 | 44.1% | 38.2% |
| 49–59% | 55 | 53.8% | 47.3% | 49–57% | 55 | 52.8% | 41.8% |
| 60–72% | 55 | 66.3% | 60.0% | 57–67% | 55 | 62.4% | 60.0% |
| 72–91% | 55 | 79.7% | 70.9% | 67–89% | 55 | 75.0% | 76.4% |

Both models are mostly too kind to the first-pick team: in the test period it won only 47.6%, against 52.5% in training. The cause is the 2026 side rule. In test, 151 of 275 first-pick teams played on red, and blue won 56.7% of test games. In training only 92 first-pick teams were on red, so the model had little chance to learn the side effect apart from first pick.

### By draft stage (test, main model)

| Draft actions done | Accuracy | Log loss |
|---|---|---|
| 0 (no draft yet) | 65.8% | 0.6380 |
| 6 (first bans) | 66.2% | 0.6352 |
| 12 (first 3 picks each) | 64.7% | 0.6345 |
| 16 (all bans) | 65.1% | 0.6331 |
| 20 (full draft) | 65.1% | 0.6312 |

Log loss improves steadily as the draft fills in, but only slightly.

### Sides

| | Train (before 2026-04) | Test |
|---|---|---|
| First-pick team win rate | 52.5% | 47.6% |
| Blue side win rate | 52.0% | 56.7% |
| First-pick team on red | 92 games | 151 games |

## Live use: `predict_draft`

```python
from draft_analyzer.model import predict_draft

r = predict_draft("Cloud9 Kia", "LYON", side_a="blue",
                  picks_a=["Ryze"], picks_b=["Cassiopeia", "Ziggs"],
                  bans={"a": ["Varus"], "b": ["Azir"]})
r["p_team_a"]            # P(Cloud9 wins), given the draft so far
r["p_team_a_elo_only"]   # the same from team strength alone
r["contributions"]       # push of each feature on the log-odds, from Cloud9's side
r["actions"]             # each pick/ban with the sample sizes behind it
```

- **Inputs:**
  - Teams by name or GRID id; champions by name or id.
  - 0–5 picks per side. Bans are a flat list, or `{"a": [...], "b": [...]}` when you know who banned what (only then do they feed `d_denied`).
  - `date` sets the history cutoff (only games before it are used). The default is the day after the last game in the data. `patch` ends the champion window at that patch.
  - `first_pick` is "a" or "b". The default is the blue team.
- **Checks:** it raises a clear error for unknown teams or champions, more than 5 picks, or a champion used twice.
- **Sample sizes:** returned for every number, as AGENTS.md rule 4 requires: `champ_picks`/`champ_wins` in the window, `window_games`, `team_games_on_champ`/`team_wins_on_champ`, `team_games`, and `history_games`.
- **The saved model** is refitted on **all** games with the same settings. The validation numbers above come from the version trained before 2026-04-01.

## Tests

`tests/test_model.py`. The tests use a small made-up drafts table, so they run without the licensed data:

- **No leakage:** features for game N are identical when every game after N is removed (checked at three cut points), and when the results of all games on game N's date are flipped. A deliberately injected leak makes both tests fail.
- **Partial drafts:** valid probabilities for 0v0, 1v0, 1v2, 3v2 and 5v5 picks. P(A) + P(B) = 1, and swapping the teams gives exactly 1 − p. `predict_draft(date=D)` ignores games on or after D. Bad drafts are rejected.
- **Real data:** the no-leakage test on the real drafts table. It is skipped when `data_cache/drafts.parquet` is missing.

## Limitations

- **The draft signal is weak and uncertain.** Treat the live win chance as "team strength, nudged by the draft", not as a verdict on the draft.
- **Small test set.** With 275 games, accuracy is only good to about ±5.6 points.
- **Season changes.** Elo carries over between seasons with no reset, though rosters change and LTA teams moved to the LCS in 2026. The start of a season is predicted worst (inner validation was close to a coin flip).
- **Side rule shift.** The 2026 first-selection rule makes "first pick" and "blue side" separate things. Training has few such games, so the side effect is under-learned (see Calibration).
- **Unknowable inputs.** Who held first selection, and why they chose first pick or a side, is not in the data.
- **Roles are inferred** (see README). Comfort uses the role the team usually plays a champion in, not the role in that game, because the role isn't known live.
- **Fixed settings.** Champion window, smoothing and the comfort window were fixed in advance, not tuned. Other choices might do better, but trying many on 175 validation games would mostly fit noise.
- **No player-level features.** Comfort is per team, so a roster change resets nothing.
- **Not causal.** Coefficients describe associations in pro games. For example, the negative ban feature most likely reflects who makes those bans, not their effect.
- **Leagues aren't comparable.** Elo is only comparable within a league, since teams from different leagues barely play each other in this data.
