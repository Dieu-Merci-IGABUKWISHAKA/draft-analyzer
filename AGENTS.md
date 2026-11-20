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
  - **Columns:** `series_id, game_id, game_number, event, stage, date, patch, team_id, team_name, opponent_id, opponent_name, side, action, draft_order, champion_id, champion_name, game_winner_team_id`, plus `player_name, role_inferred, role_source` on picks.
  - **Commands:** `load_drafts(rebuild=False)` reads the cache. `python src/draft_analyzer/drafts.py --rebuild` rebuilds it.
- **`roles.py`**: infers each player's role (top, jungle, mid, bot, support), because the data has no role field. In 6 early-2025 events, roster order is role order; those games are the labels. Every other team-game gets the one-player-per-role assignment that fits both the player's usual role and the champion. Holding out each labelled event in turn, it was right for 100% of player-games.
- **`analysis.py`**: every function takes a GRID `team_id` and returns a DataFrame that includes sample sizes.
  - **Recency, on most functions:** `last_patches=N` keeps the team's N most recent patches. `half_life=H` weights a game by `0.5 ** (patches_ago / H)`.
  - **Records and window:** `team_games`, `record`, `window_patches`, `data_summary`, `team_options`.
  - **Tendencies:**
    - `champion_pool`: picks per role, with W-L and recency.
    - `priority_picks`: the very first pick and first-round picks (slots 7–12), split by whether the team had first pick.
    - `bans_by`: bans in phase 1 vs phase 2.
    - `bans_against`: bans aimed at the team. Both ban functions take `vs=` for head-to-head only.
  - **Ban plan:** `recommend_bans` with `DEFAULT_BAN_WEIGHTS` (every knob in one dict). The score is the opponent's pick rate × their smoothed win-rate edge over 50%, plus a small term for how often other teams ban it against them. Positive scores are cut when we play the champion ourselves. Win rates are smoothed with 10 imaginary games at the opponent's overall rate, and champions with fewer than 4 picks are skipped. Each pick comes with plain-English `reason_lines`.
  - **Fearless:**
    - `fearless_events` detects the rule per event from repeated picks within a series ("hard" means no team ever re-picks).
    - `fearless_game_ids` lists the games in multi-game Fearless series.
    - `fearless_by_game` gives record, distinct champions and top picks for buckets **G1 / G2 / G3+**.
    - `series_plan` takes the champions already used and returns what the opponent loses, their likely picks now, and new bans.

- **`model.py`**: draft win-probability model, P(first-pick team wins), described in [docs/MODEL.md](docs/MODEL.md).
  - **Features:** Elo, side, and draft features summed over picks and bans so far, first-pick team minus the other. Every feature uses only games on **earlier dates**.
  - **Training:** each game appears at all 21 draft stages, so the model works mid-draft.
  - **Choice and validation:** K is tuned on 2025 games. The model is chosen on 2026-01-01..03-31 and tested once on games from 2026-04-01.
  - **Commands:** `python src/draft_analyzer/model.py` validates and saves the model. `load_model()` loads it. `predict_draft(team_a, team_b, side_a, picks_a, picks_b, bans, date, patch)` scores a partial draft and returns sample sizes for every number.

### `app/streamlit_app.py`

`st.cache_data` wrappers around the `analysis` functions, then layout only.
- **Sidebar:** matchup (us and the opponent, default LYON), recency (last N patches, default 6, and the half-life), and a slider for each ban-score weight.
- **Header:** overall and head-to-head records.
- **Four tabs:**
  - **Opponent scouting:** champion pool chart and table, priority picks, bans by phase.
  - **Against us:** bans each side makes in head-to-head games.
  - **Ban plan:** top 5 ban cards with reasons.
  - **Fearless series:** stats by game number, plus the series planner.

Run it with `python -m streamlit run app/streamlit_app.py`.

### Other files

- `notebooks/00_check_data.py`: a smoke test that the data is reachable. It streams one Cloud9 feed.
- `notebooks/01_cloud9_scouting.py`: a text version of the scouting report.
- `.idea/runConfigurations/`: shared PyCharm configurations: **Dashboard (Streamlit)**, **Rebuild drafts table**, **Cloud9 scouting report**, **Train draft model**, **Tests**.
- `docs/AI_WORKFLOW.md` (how the project was built with Claude Code), `docs/SUBMISSION.md` (Devpost text), `docs/MODEL.md` (win-probability model and validation), `docs/images/` (screenshots, still to add).
- `tests/`: pytest (`python -m pytest`, or the **Tests** run configuration). The tests build a synthetic drafts table, so they run without the licensed data. The real-data leakage test is skipped when the cache is missing. Test data must stay synthetic: never copy GRID rows into a fixture.

## Existing findings

From data as of patch 16.17. The README holds the current wording, so keep it in sync.

- **Data quality.** The event stream and the final snapshot agree in 1,734 of 1,739 games. In the other 5 the stream misses bans. The stream also has 461 undo events (44 abandoned draft attempts).
- **2026 first selection.** 243 games in 2026 have the first-pick team on red side.
- **Hard Fearless everywhere.** All 20 events use hard Fearless Draft. In 568 multi-game series no champion is ever picked twice, by either team. This is detected from the picks, not assumed.
- **Cloud9 Kia vs LYON.** Cloud9 is 4–9 against LYON in the last 6 patches (13 games). In the same window LYON is 27–12 and Cloud9 23–22. Over all data Cloud9 is 90–52.
- **Respect ban vs Cloud9 has shifted.** A respect ban removes a champion because the other team is strong on it. Ziggs is the most-banned champion against Cloud9 overall (41 bans in 142 games). With recency weighting (half-life 3 patches), Cassiopeia leads at 39% and Ziggs drops to 17%.
- **LYON by game number in Fearless series, last 6 patches.** LYON is 11–2 in game 2, against 8–5 in both game 1 and game 3+ (13 games each). Over all data the gap is small (53% / 60% / 59%), so treat this as a recent trend on a small sample.
- **Win-probability model** (test: 275 games from 2026-04-04 to 2026-10-13, trained on earlier games):
  - Elo + side + draft gets 65.1% accuracy and log loss 0.631, against 64.4% and 0.642 for Elo alone.
  - The draft's gain is small and not statistically clear: log loss +0.011, 95% CI −0.002..+0.025.
  - A full draft moves the estimate by a median of 2.3 points.
  - In 2026 blue side won 56.7% of test games, and first-pick teams only 47.6%, because many first-pick teams played on red.
- **Ban score tuning history.** Version 1 over-ranked Nocturne, a champion LYON loses on (44%). Version 2 put LeBlanc (3–0) and Varus (4–0) on top. The fixes were stronger smoothing and `min_picks`. Now Ryze (6–1 in 7 games) outranks Varus (4–0). These weights were set by judgment on all the data, **not** checked on held-out later games (see rule 5).

## Status vs the judges' questions

| Ask | Status |
|---|---|
| Repeated pick/ban tendencies | **Built.** Opponent scouting and Against us tabs, plus the Ban plan. |
| Fearless G3, G4, G5 shift | **Partly built.** The code merges games 3–5 into one "G3+" bucket. Judges ask about G3, G4 and G5 separately: show them apart, even if the sample is small (flag it), and compare to G1 and G2. |
| Split Fearless "G3+" into G3, G4 and G5 | **Planned for Phase 2.** Each game number gets its own row with its sample size (games and W-L), and small samples are flagged. |
| "How to win" insights | **Not built.** |
| Live draft win-rate impact | **Model built, not in the app yet.** `model.predict_draft` works on partial drafts and was validated on later games (docs/MODEL.md). Still to do: a dashboard panel that updates as picks are entered and shows the held-out accuracy and the sample sizes. Be honest on screen that the draft moves the estimate only a few points. |
| Time-based validation | **Done for the win-probability model only.** The role inference was checked by holding out whole events, not later games. The ban-score weights were not validated on held-out games. |
| Screenshots and demo video | **Not done.** `docs/images/` is empty, and the video link in the README is a placeholder. |
