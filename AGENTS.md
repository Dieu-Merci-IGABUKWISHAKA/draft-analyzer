# AGENTS.md: project brief

Every task in this repo, by a person or an AI agent, follows this file. If a request conflicts with a hard rule below, stop and say so instead of working around it.

## Goal

Win **Best of Category 2 (LoL Macro Scouting & Draft Management)** in the Cloud9 x JetBrains **"Sky's The Limit"** hackathon.

Judging criteria, and what each one means for our work:

| Criterion | What judges look for | What it means for us |
|---|---|---|
| **Technological Implementation** | Depth of JetBrains tool use | Build in PyCharm and make that visible: shared run configurations, committed `.idea/`, inspections, the debugger, the PyCharm MCP server. Do not just use it as a text editor. |
| **Design** | A complete, coherent product | One dashboard a coach can use from start to finish. Every number is explained, nothing is half-built, no dead ends. |
| **Potential Impact** | Solves a real coach problem | Prepare the draft against a specific opponent, including in a Fearless series. |
| **Quality of Idea** | Novelty | Things other entries won't have, for example a Fearless series planner, Fearless rules detected from the data, ban reasons in plain English, and checked predictions. |
| **Presentation** | The demo video | Each feature has to show up clearly on screen, with a real finding behind it. |

## Questions the product must answer visibly

These are the judges' own questions. Each needs its own clearly labelled place in the dashboard, so a judge can find the answer without looking for it.

1. **"What pick/ban tendencies repeat most often for this opponent?"**
2. **"How does the opponent's draft strategy shift in a Fearless Draft Game 3, 4 or 5?"**

Next-level asks (what moves us from good to winning):

3. **Data-backed "how to win" insights**: concrete, testable advice, for example "they lose when denied X early", with the record and sample size behind it.
4. **Live draft win-rate impact**: while a draft is being entered pick by pick, show how each pick or ban changes the estimated win chance.

Current coverage: see [Status vs the judges' questions](#status-vs-the-judges-questions).

## Hard rules

1. **The GRID data is licensed. Never commit it.** That includes anything derived from it at row level: `.parquet`, `.duckdb`, `.sqlite`, `.csv` extracts, `.jsonl`, `.zst`, copies of `meta/`, and anything in `data_cache/`. Only **code, charts, screenshots and aggregate numbers** may be committed. Do not upload the data or extracts anywhere else either (paste sites, cloud notebooks, public deploys). The data lives in `../stl-hackathon-2026-grid-starter` (or `$STL_DATA_DIR`); its `LICENSE.md` and `AGENTS.md` apply.
2. **No Riot artwork.** No champion images, icons, splash art, or Riot logos. The product is text and charts only.
3. **All logic goes in `src/draft_analyzer/`.** `app/streamlit_app.py` only lays out what the package returns: no statistics, filtering rules, or scoring in the app. Notebooks only call the package too.
4. **Every statistic shows its sample size.** Rates come with games or a W-L record. Small samples (fewer than `SMALL_SAMPLE` = 5 games) are flagged or greyed out, never hidden silently.
5. **Validate on later games than the ones used to fit or tune.** Any model, weight, threshold or prediction is fitted or tuned on earlier games and judged on later ones: split by date or patch, never randomly. Nothing from the test period may leak into features, priors, smoothing, or the choice of weights. Report the result on the held-out games with its sample size.
6. **Run the data check before every commit.** From the repo root (Git Bash):

   ```bash
   git ls-files | grep -Ei '\.(zst|jsonl|parquet|sqlite|duckdb|csv|joblib|pkl)$|stl-hackathon-2026-grid-starter|data_cache/'
   ```

   It must print nothing. Also read `git diff --cached --name-only` before committing, and stage files by name, not with `git add -A` or `git add .`. If data was ever committed, stop and tell the user. Deleting it in a new commit is not enough because it stays in history.

Working conventions:
- Explain draft terms and metrics in plain words (the user is learning LoL). Define jargon the first time it appears in the UI or docs.
- Check a claim against the data before building on it, and update the README findings when a number changes.
- Commit messages end with the co-author line the session gives you.

## Architecture

```
GRID feeds (.jsonl.zst, outside the repo) + meta/*.csv
        │  src/draft_analyzer/drafts.py   (build once, ~40 s)
        ▼
data_cache/drafts.parquet   (gitignored, one row per pick or ban)
        │  src/draft_analyzer/analysis.py (metrics, ban score, Fearless)
        │  src/draft_analyzer/model.py    (win-probability model → data_cache/draft_model.joblib)
        ▼
app/streamlit_app.py (dashboard)    notebooks/01_cloud9_scouting.py (text report)
```

### `src/draft_analyzer/`

- **`config.py`**: paths. `DATA_DIR` comes from `STL_DATA_DIR`, or defaults to `../stl-hackathon-2026-grid-starter`. `CACHE_DIR` = `data_cache/`. `CLOUD9_LOL_TEAM = "Cloud9 Kia"`.
- **`drafts.py`**: builds the drafts table.
  - **Source of truth:** the final `seriesState.games[].draftActions` of each feed, not the event stream. The snapshot already resolves undo events and rollbacks, and it has 5 games' worth of bans that the stream misses.
  - **Joins:** patch, sides, team names and the winner come from `meta/games.csv`, and champion names from `meta/characters.csv`, always on IDs.
  - **Draft slots:** `draft_order` is the standard slot 1–20 (bans 1–6, picks 7–12, bans 13–16, picks 17–20), counted from the team that picks first, not from blue side. Since 2026 ("first selection") that team is often red. `_slots_by_occurrence` handles feeds logged out of order.
  - **Columns:** `series_id, game_id, game_number, event, stage, date, patch, team_id, team_name, opponent_id, opponent_name, side, action, draft_order, champion_id, champion_name, game_winner_team_id`, plus `player_id, player_name, role_inferred, role_source` on picks.
  - **Objective events:** the same pass (`scan_series`) collects first-blood, dragon, grub, herald and tower events, with their game-clock time, into `data_cache/objective_events.parquet` for macro.py.
  - **Commands:** `load_drafts(rebuild=False)` reads the cache. `python src/draft_analyzer/drafts.py --rebuild` rebuilds it (about 40 s).
- **`roles.py`**: infers each player's role (top, jungle, mid, bot, support), because the data has no role field. In 6 early-2025 events, roster order is role order; those games are the labels. Every other team-game gets the one-player-per-role assignment that fits both the player's usual role and the champion. Holding out each labelled event in turn, it was right for 100% of player-games.
- **`analysis.py`**: every function takes a GRID `team_id` and returns a DataFrame that includes sample sizes.
  - **Recency, on most functions:** `last_patches=N` keeps the team's N most recent patches. `half_life=H` weights a game by `0.5 ** (patches_ago / H)`.
  - **Records and window:** `team_games`, `record`, `window_patches`, `data_summary`, `team_options`.
  - **Tendencies:**
    - `champion_pool`: picks per role, with W-L and recency.
    - `priority_picks`: the very first pick and first-round picks (slots 7–12), split by whether the team had first pick.
    - `bans_by`: bans in phase 1 vs phase 2.
    - `bans_against`: bans aimed at the team. Both ban functions take `vs=` for head-to-head only.
  - **Ban plan:** `recommend_bans` with `DEFAULT_BAN_WEIGHTS`, tuned on 2025 games (see backtest.py).
    - **Score:** recent pick rate × (2 + 3 × smoothed win-rate edge over 50%), plus 1 × how often other teams ban it against them.
    - **Filter:** champions with fewer than 2 picks are skipped.
    - **Reasons:** each pick comes with plain-English `reason_lines` ordered by their contribution to the score (`drivers`), so the first line is the driving reason. `opponent_name` sets the wording.
  - **Fearless:**
    - `fearless_events` detects the rule per event from repeated picks within a series ("hard" means no team ever re-picks).
    - `fearless_game_ids` lists the games in multi-game Fearless series.
    - `fearless_by_game` gives record, distinct champions and top picks for each game number **G1 to G5**, with `small_sample` flags.
    - `series_plan` takes the champions already used and returns what the opponent loses, their likely picks now, and new bans.

- **`model.py`**: draft win-probability model, P(blue side wins), described in [docs/MODEL.md](docs/MODEL.md).
  - **Features:** Elo, side (the intercept) and `blue_first_pick` as separate features. Draft features are blue minus red, summed over the picks and bans so far: meta win rate, presence, first-phase priority, team and player comfort, lane matchups, and bans. Every feature uses only games on **earlier dates**.
  - **Training:** each game appears at all 21 draft stages, so the model works mid-draft.
  - **Validation:** walk-forward by month. Each month is trained on all earlier games. Its C and Platt calibration come from earlier months' out-of-sample predictions only. Elo K is tuned before July 2025. `evaluate()` reports results with series-bootstrap CIs, per league, plus permutation importance.
  - **Commands:** `python src/draft_analyzer/model.py` validates and saves the model. `load_model()` loads it.
  - **`predict_draft`:** `predict_draft(team_a, team_b, side_a, picks_a, picks_b, bans, date, patch, roles_a, roles_b)` scores a partial draft. It returns the calibrated win chance, each feature's push, and the sample size behind every number.

- **`simulator.py`**: the Draft simulator, described in [docs/SIMULATOR.md](docs/SIMULATOR.md).
  - **Inputs:** `Setup` (side, first pick, game number, Fearless locks, lineups) plus a tuple of `Action`s.
  - **`Simulator(our_id, opp_id, live)` methods:** `apply`, `set_role`, `win_chance` (with a bootstrap band), `trajectory`, `best_options` and `likely_next`.
  - **Bans:** they are valued by the pick they deny. They are not fed into the win model.
  - **Tendencies:** `tendency_tables`, `pick_probs` and `ban_probs` describe the opponent's habits, with fixed weights. `likely_accuracy` checks them walk-forward.
  - **Speed:** the feature cache per draft prefix keeps each step under 1 s.

- **`insights.py`**: insight cards, described in [docs/INSIGHTS.md](docs/INSIGHTS.md). They are built on records and draft habits, not on the win model.
  - **`Scout(drafts)`:** builds cards for any team, in opponent or self view. The card types are ban impact, Fearless carry-over, comfort dependency, side, first pick, and phase tells.
  - **Statistics:** Wilson 90% intervals, two-proportion confidence, and a minimum of 8 games per group.
  - **Predictability:** entropy and top-3 share of first picks and first bans, compared with the league.
  - **Replication:** `replication` and `replication_summary` re-measure each kind of pattern on later games (splits July 2025 and April 2026).
  - **Ranking:** `report` ranks by gap / typical gap × confidence × kept, where "kept" is the share of the kind's confident gaps that survived. "How to beat" needs confidence ≥ 80% and kept ≥ 25%.

- **`macro.py`**: early-game context, described in [docs/MACRO.md](docs/MACRO.md).
  - **`first_objectives`:** the first blood, dragon, void grub, herald and tower per game: team and game time. Undo events are applied, and towers are credited by the owner side.
  - **`Macro(scout)`:** `profile` gives objective rates and median times against the league. `jungler_links` gives the team's rate with one jungler against its other junglers.
  - **Replication:** `replication` and `replication_summary` re-measure each pattern on later games.
  - **`summary`:** returns only rows whose kind holds up (`insights.holds_up`: kept ≥ 25% and the later gap's 90% interval above zero) and that are confident (≥ 80%).

- **`backtest.py`**: walk-forward backtest of the Ban plan, described in [docs/VALIDATION.md](docs/VALIDATION.md).
  - **`build_cases`:** the per-game Ban plan inputs, from earlier games only, with Fearless locks. Cached in data_cache.
  - **Lists:** `plan_lists` uses `analysis.ban_stats` and `ban_scores` (vectorised), compared against naive and meta baselines.
  - **Metrics and tuning:** `evaluate` measures pick rate, game hits and win checks; `paired_gain` gives bootstrap CIs; `tune` runs a 540-setting grid on 2025.
  - **Previous weights:** `PREVIOUS_BAN_WEIGHTS` keeps the old hand-set weights for comparison.
- **`analysis.recommend_bans`** is split into `ban_stats` (slow, weight-independent) and `ban_scores` (cheap). Its output is unchanged.

- **`prep.py`**: Match prep. `match_prep` assembles the one-screen brief from existing functions, with no new statistics. `render_html` produces the printable, text-only pre-match report.

### `app/streamlit_app.py`

`st.cache_data` wrappers around the `analysis` functions, then layout only.
- **Theme and sidebar:** `.streamlit/config.toml` sets the theme (calm blue primary, sans serif). The sidebar opens with a 2-line intro, then the matchup (us and the opponent, default LYON), recency (last N patches, default 6, and the half-life), and a slider for each ban-score weight.
- **Header:** overall and head-to-head records.
- **Seven tabs, in this order:** Match prep, Draft simulator, Opponent scouting, Insights, Ban plan, Fearless series, Against us. The Match prep code sits at the end of the script so it can reuse every cached loader.
  - **Match prep:** head-to-head and form, what they'll draft, our ban plan, how to beat them, and watch out (their bans vs us, our habits), plus the proof line and the "Export pre-match report" download. Every block names its detail tab.
  - **Opponent scouting:** champion pool chart and table, priority picks, bans by phase, and a short **Early game** section (macro.py, 12 patches): only lines that held up, plus an expander with every number and the replication table.
  - **Against us:** bans each side makes in head-to-head games.
  - **Ban plan:** top 5 ban cards, strongest reason first, with the backtest caption (`backtest.headline_caption`).
  - **Fearless series:** stats for games 1 to 5, plus the series planner.
  - **Insights:** "How to beat <opponent>" (top 5 cards that pass the bar), their draft habits, all patterns, "Are these patterns real?" (likely-pick accuracy plus the replication table), and self-scouting for our team.
  - **Draft simulator:** the board, win chance with its band, best options, and likely picks from them. Session state holds the actions; the draft is replayed through `Simulator.apply` on every rerun.

Run it with `python -m streamlit run app/streamlit_app.py`.

### Other files

- `notebooks/00_check_data.py`: a smoke test that the data is reachable. It streams one Cloud9 feed.
- `notebooks/01_cloud9_scouting.py`: a text version of the scouting report.
- `.idea/runConfigurations/`: shared PyCharm configurations: **Dashboard (Streamlit)**, **Rebuild drafts table**, **Cloud9 scouting report**, **Train draft model**, **Tests**.
- `docs/AI_WORKFLOW.md` (how the project was built with Claude Code), `docs/SUBMISSION.md` (Devpost text), `docs/MODEL.md` (win-probability model and validation), `docs/SIMULATOR.md` (draft simulator), `docs/INSIGHTS.md` (insight cards and their replication), `docs/MACRO.md` (early-game patterns and their replication), `docs/VALIDATION.md` (**every validation result in one place**), `docs/images/` (screenshots, still to add).
- `tests/`: pytest (`python -m pytest`, or the **Tests** run configuration). The tests build a synthetic drafts table, so they run without the licensed data. The real-data leakage test is skipped when the cache is missing. Test data must stay synthetic: never copy GRID rows into a fixture.

## Existing findings

From data as of patch 16.17. The README holds the current wording, so keep it in sync.

- **Data quality.** The event stream and the final snapshot agree in 1,734 of 1,739 games. In the other 5 the stream misses bans. The stream also has 461 undo events (44 abandoned draft attempts).
- **2026 first selection.** 243 games in 2026 have the first-pick team on red side.
- **Hard Fearless everywhere.** All 20 events use hard Fearless Draft. In 568 multi-game series no champion is ever picked twice, by either team. This is detected from the picks, not assumed.
- **Cloud9 Kia vs LYON.** Cloud9 is 4–9 against LYON in the last 6 patches (13 games). In the same window LYON is 27–12 and Cloud9 23–22. Over all data Cloud9 is 90–52.
- **Respect ban vs Cloud9 has shifted.** A respect ban removes a champion because the other team is strong on it. Ziggs is the most-banned champion against Cloud9 overall (41 bans in 142 games). With recency weighting (half-life 3 patches), Cassiopeia leads at 39% and Ziggs drops to 17%.
- **LYON by game number in Fearless series, last 6 patches.** LYON is 11–2 in game 2, against 8–5 in both game 1 and game 3+ (13 games each). Over all data the gap is small (53% / 60% / 59%), so treat this as a recent trend on a small sample.
- **Win-probability model** (walk-forward test: 897 games in 13 monthly folds, July 2025 to October 2026; each month trained on earlier games only):
  - **Elo alone:** 64.1% accuracy, log loss 0.639.
  - **Full model with the draft:** 65.2% accuracy, log loss 0.645.
  - **The draft gain is not statistically real.** Full vs Elo + side: log loss −0.004 (95% CI −0.010..+0.003), accuracy −0.1 pts (CI −2.0..+1.7). No league shows a reliable gain either.
  - **Elo dominates.** Permutation importance is 0.111 for Elo vs 0.003 for all draft features together. In the draft, only team and player comfort carry weight, and they correlate with Elo (r 0.27–0.36).
  - **What a draft shifts:** a full draft moves the estimate by a median of 3.0 points (90% under 7.6).
  - **Sides:** blue won 52.3% of test games. First pick is no advantage: blue won 51.8% with first pick and 53.5% without it (243 games).
  - **Calibration:** Platt calibration did not help (log loss 0.641 → 0.645). The final calibration is nearly the identity.
  - **Phase 1 is superseded:** its single-split result (275 games, draft gain +0.011 log loss) did not hold up.
- **Simulator "likely from them"** (walk-forward, 897 games, each prediction using only earlier games):
  - **Picks:** the actual pick was in the top 5 44% of the time (league meta alone: 37%), and the top guess 13% of the time.
  - **Bans:** top 5 38% of the time (meta alone: 26%).
- **Which insight patterns hold up in later games** (confident patterns, two split dates pooled):
  - **Phase tells:** 82% point the same way later, keeping 57% of the gap.
  - **Comfort dependency:** 82%, keeping 48% (11 checks).
  - **Ban impact:** 75%, keeping 30% (24 checks). Across all ban gaps, including weak ones, only 52%.
  - **Side, first pick and Fearless carry-over:** 20–50%, keeping 0%. Treat them as anecdotes.
- **Early-game patterns that hold up in later games** (confident patterns, two splits pooled):
  - **First-tower rate:** 74% same direction, keeping 53% of the gap. This mostly reflects team strength.
  - **Jungler–first-grub link:** 74%, keeping 27%. It only just passes (later-gap CI +1 to +18 pts).
  - **Everything else fails:** first blood, dragon and herald rates; jungler links for other objectives; all timings.
- **Feed quirk:** early-2025 feeds have no generic `slayDragon` event, only the drake-type events.
- **Winner's curse:** in a test draft vs LYON where Cloud9 always took the top-ranked option, the estimate climbed from 40% to 51%. The best of many noisy estimates is too high, so the simulator says so on screen.
- **Ban score tuning history.** Version 1 over-ranked Nocturne, a champion LYON loses on (44%). Version 2 put LeBlanc (3–0) and Varus (4–0) on top. The fixes were stronger smoothing and `min_picks`. Now Ryze (6–1 in 7 games) outranks Varus (4–0). Those hand-set weights were later backtested (Phase 5) and were no better than the league meta in 2026 (26.5% vs 26.9% pick rate).
  - **Weights tuned on 2025 games are now the defaults:** pick 2, banned_by_others 1, our_pick_penalty 0, min_picks 2.
  - **2026 result:** 33.8% of the listed, open champions picked, against 19.8% naive and 26.9% meta.
  - **Against LYON** the top ban is now Varus: 4–0 on a small sample, but banned against LYON in 16 games. "Banned by other teams" turned out to be predictive.

## Status vs the judges' questions

| Ask | Status |
|---|---|
| Repeated pick/ban tendencies | **Built.** Opponent scouting and Against us tabs, plus the Ban plan. |
| Fearless G3, G4, G5 shift | **Built (Phase 2).** The Fearless tab shows games 1 to 5 separately, each with its record and game count. Games with fewer than 5 games are flagged, and games 4–5 carry a caption that they come only from best-of-5 series. The simulator also takes the game number and the champions locked by earlier games. |
| "How to win" insights | **Built (Phase 3): Insights tab.** "How to beat <opponent>" lists the top 5 cards, with their evidence, Wilson 90% intervals, confidence and sample sizes. It is backed by a replication check on later games, and only kinds of pattern that held up qualify. Self-scouting runs the same analysis on our team, plus how predictable our first picks and bans are. |
| Live draft win-rate impact | **Built (Phase 2): Draft simulator tab.** It shows our win chance with a 90% band beside Elo only, a chart of how it moved, the top 5 options with their shift, reason and sample sizes, and the opponent's likely next picks with their checked accuracy. It is framed as "what each choice shifts", never as a forecast. It was checked in the browser for LYON and FlyQuest. |
| Time-based validation | **Done for the model, likely picks, insight and macro patterns, and the Ban plan** (docs/VALIDATION.md). Not time-based: role inference (held out by whole events, not later games). |
| Early-game macro context | **Built (Phase 4).** Five first objectives per game, with team and time, extracted in the drafts pass. The Opponent scouting tab shows only lines that held up in later games (docs/MACRO.md). Most early-game habits did not. |
| Ban plan validated | **Done (Phase 5).** Walk-forward backtest with weights tuned on 2025 and tested on 2026. It beats the naive top-5 by +14.0 pts and the league meta by +6.9 pts on pick rate. There's no consistent evidence the listed champions win them more games. See docs/VALIDATION.md and the README "Does it work?" section. |
| Finished-product pass | **Done (Phase 6).** Match prep first tab, HTML export, theme, sidebar intro, ordered ban reasons and the backtest caption. Checked in Chrome for LYON, FlyQuest, Team Liquid Alienware and G2 Esports, with no console or server errors. The narrow-width visual check is not done (screenshots failed after a window resize). |
| Screenshots and demo video | **Not done.** `docs/images/` is empty, and the video link in the README is a placeholder. |
