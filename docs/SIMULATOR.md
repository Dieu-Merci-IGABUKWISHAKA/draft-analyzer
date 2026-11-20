# Draft simulator

The **Draft simulator** tab lets a coach play a draft pick by pick against the chosen opponent. At each step it shows what each choice **shifts**. The logic is in `src/draft_analyzer/simulator.py`; the app only lays it out.

**How to read it:** in walk-forward testing the draft added no reliable accuracy beyond team strength (see [MODEL.md](MODEL.md)). A whole draft moves the estimate by about 3 points. So the numbers are **nudges backed by past games**, each with its sample size, not forecasts of who wins.

## The board

- **Draft order:** the real pro order, with bans 1–6, picks 1–6, bans 7–10, then picks 7–10. The order is keyed on **which team picks first**. Our **side** is set separately, because since 2026 the team with first selection chooses either side or first pick.
- **Making a move:** pick a champion in the searchable selector, or press **Pick X / Ban X** on a suggestion. Undo and Reset are next to it.
- **Roles:** each pick's role defaults to the role this team usually plays the champion in, and can be changed on the board. If a teammate already has that role, the two swap.
- **Players** come from each team's last lineup and can be changed under *Lineups*.
- **Fearless:** set the game number (1–5) and enter the champions used earlier in the series. They are locked for both teams (hard Fearless, which all 20 events in the data use).
- **Changing settings mid-draft:** the draft is replayed under the new settings. If a move becomes illegal (for example a champion that is now locked), the draft is cut back to just before it, with a warning.

## What it shows

**Win chance:**
- The calibrated model estimate for our team, with a **90% range**. The range is the spread across 50 refits of the model on resampled series, i.e. how sure the model is of its own estimate; any single game is far less certain.
- Next to it, the estimate from **team strength (Elo) alone**, and a line chart of how the estimate moved during the draft.

**Best options for us (our pick):**
- **The candidates:** every available champion that fits one of our open roles, ranked by how much it moves our win chance.
- **For each, the tab shows:**
  - the shift in points, with its 90% range across the 50 refits;
  - the **main reason**: the group of model features that pushed it most, which is player comfort, team comfort, lane matchup, or meta strength;
  - the **evidence**, for example "Thanatos: 10 games on it (9–1)" or "We played it as jungle 3× (3–0 · small sample)".

**Best options for us (our ban):**
- **How a ban is scored:** a ban is valued by the pick it denies. The value is the chance they pick it later in this draft (`their remaining picks × chance it is their next pick`), multiplied by how much worse it is for us than their typical pick.
- **Why bans don't feed the win model directly:** the model's ban feature is an association, not an effect. Underdogs tend to ban the favourite's comfort picks. Feeding bans in would tell the simulator that banning the opponent's best champion hurts you.

**Likely from them:**
- **What it shows:** the opponent's 5 most likely choices for their next pick or ban, with the chance and the evidence.
- **Picks blend:**
  - their pick habits in this draft phase over the last year (recent patches weigh more; picks from the other phase count 30%), weighted 70%;
  - the league meta, weighted 30%.
  - Candidates must fit one of their open roles and still be available.
- **Bans blend:** their ban habits in this phase (50%), the league meta (25%), and what our team plays (25%).
- **The weights were fixed in advance, not tuned.**

## Was "likely from them" any good?

Checked walk-forward on 897 games from July 2025 to October 2026. Each prediction used only games from earlier dates, plus the Fearless locks from earlier games in the series:

| | Predictions | Actual choice was the top guess | In the top 5 | League meta alone: top 5 |
|---|---|---|---|---|
| Picks | 8,970 | 13% | **44%** | 37% |
| Bans | 8,964 | 11% | **38%** | 26% |

That beats the meta alone, especially for bans. Still, more often than not the actual choice is **not** in the top 5, so the list is a briefing aid, not a prediction. Rerun the check with `python src/draft_analyzer/simulator.py` (about a minute; it caches aggregate numbers only in `data_cache/`).

## Limitations

- **Winner's curse:** taking the top-ranked option every time overstates the gain, because the best of ~100 noisy estimates is usually too high. In a test draft where Cloud9 always took the top option, the estimate climbed from 40% to 51%. Treat a gap of a point or two between options as a tie.
- **Uncertainty band:** it covers the model's uncertainty about its own estimate, not the much larger uncertainty of a single game.
- **What the model can't see:** recent scrims, patch notes that came after the data, and player form.
- **Roles and players are guesses** until the coach sets them. Flex picks and substitutes need a manual check.
- **No synergy:** champion synergy within a team isn't modelled (the data is too thin). Only lane-vs-lane matchups are.

## Performance

- **Per click:** each step (win chance, chart, best options, likely picks) takes **50–450 ms** on the real data.
- **How:** the features of each draft prefix are cached, so a click only computes the newest step.
- **Startup:** loading the history and model takes about 5 s once per app session.
- **Tests:** `tests/test_simulator.py` (synthetic data) covers:
  - the draft order;
  - the rules (Fearless locks, duplicates, filled roles);
  - role swaps;
  - that a pick's displayed shift equals the change after applying it;
  - side symmetry;
  - that the tendencies use only earlier games;
  - that a step stays under 1 second.
