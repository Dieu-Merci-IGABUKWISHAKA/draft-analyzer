"""Match prep: the one-screen pre-match brief, and its printable HTML export.

Everything here is assembled from existing analyses; there are no new statistics:
  records / form       analysis.record, analysis.team_games
  what they'll draft   simulator.Simulator.likely_first_phase_picks (checked walk-forward),
                       insights phase tells (a kind of pattern that held up in later games)
  our ban plan         analysis.recommend_bans (weights tuned on 2025, backtested on 2026)
  how to beat them     insights.Scout.report "top" cards (passed the bar)
  watch out            analysis.bans_by(vs=us), insights self-scouting
Each item names the tab with the details.
"""

from __future__ import annotations

import html
from datetime import date

import pandas as pd

from draft_analyzer import analysis as A
from draft_analyzer import backtest as B
from draft_analyzer import insights as I

PREP_PATCHES = 6          # records, form and the ban plan: the dashboard's default window
FORM_GAMES = 10
TABS = {"form": "Opponent scouting", "draft": "Draft simulator", "tells": "Insights", "bans": "Ban plan",
        "beat": "Insights", "vs": "Against us", "habits": "Insights (self-scouting)"}


def recent_form(team: str, drafts: pd.DataFrame, last_patches: int = PREP_PATCHES, n: int = FORM_GAMES) -> dict:
    """Record in the window and the last n results, oldest first ("W"/"L")."""
    g = A.team_games(team, drafts, last_patches).sort_values("date")
    last = g.tail(n)
    return {"games": len(g), "wins": int(g["won"].sum()), "losses": int(len(g) - g["won"].sum()),
            "last": "".join("W" if w else "L" for w in last["won"]), "last_n": len(last)}


def match_prep(our_id: str, opp_id: str, *, drafts: pd.DataFrame, scout: I.Scout, sim, insight_rep: pd.DataFrame,
               likely_acc: pd.DataFrame, weights: dict | None = None, last_patches: int = PREP_PATCHES) -> dict:
    """Everything the Match prep tab and the exported report show."""
    names = scout.team_names
    our, opp = names.get(our_id, our_id), names.get(opp_id, opp_id)
    h2h_window = A.record(our_id, last_patches, vs=opp_id, drafts=drafts)
    h2h_all = A.record(our_id, None, vs=opp_id, drafts=drafts)

    likely = sim.likely_first_phase_picks(5)
    opp_rep = scout.report(opp_id, I.DEFAULT_PATCHES, "opponent", replication=insight_rep)
    # Phase tells held up in later games; keep the confident ones not already in "How to beat them".
    beat = opp_rep["top"][:3]
    tells = [c for c in opp_rep["cards"] if c.kind == "phase tell" and c.confidence >= I.MIN_CONFIDENCE
             and c not in beat][:3]
    self_rep = scout.report(our_id, I.DEFAULT_PATCHES, "self", replication=insight_rep)
    self_tells = [c for c in self_rep["cards"] if c.kind == "phase tell" and c.confidence >= I.MIN_CONFIDENCE][:2]

    bans = A.recommend_bans(our_id, opp_id, 5, weights, drafts, last_patches, opponent_name=opp)
    vs_us = A.bans_by(opp_id, None, None, drafts, vs=our_id).head(5)
    acc = likely_acc.set_index(["kind", "method"])
    last_date = pd.to_datetime(drafts["date"]).max().date()
    return {
        "our": our, "opp": opp, "generated": str(date.today()), "data_through": str(last_date),
        "window": last_patches, "insights_window": I.DEFAULT_PATCHES,
        "h2h": {"window": h2h_window, "all": h2h_all},
        "form": {"us": recent_form(our_id, drafts, last_patches), "them": recent_form(opp_id, drafts, last_patches)},
        "likely": [{"champion": r.champion, "role": r.role, "prob": r.prob, "evidence": r.evidence,
                    "small_sample": bool(r.small_sample)} for r in likely.itertuples()],
        "tells": [_card(c) for c in tells],
        "bans": [{"champion": r.champion, "lead": r.reason_lines[0], "more": list(r.reason_lines[1:]),
                  "score": float(r.score)} for r in bans.itertuples()],
        "ban_games": int(bans["n_opp_games"].iloc[0]) if len(bans) else 0,
        "beat": [_card(c) for c in beat],
        "they_ban": [{"champion": r.champion, "bans": int(r.bans), "games": int(r.n_games),
                      "small_sample": bool(r.small_sample)} for r in vs_us.itertuples()],
        "habits": self_rep["predictability_takeaways"] + [c.takeaway for c in self_tells],
        "proof": (f"Tendencies tested on {int(acc.loc[('pick', 'tendency'), 'games'])} unseen games: next pick in our "
                  f"top 5 {acc.loc[('pick', 'tendency'), 'top5']:.0%} vs {acc.loc[('pick', 'meta only'), 'top5']:.0%} "
                  f"for the meta."),
        "ban_proof": B.headline_caption(),
        "tabs": TABS,
    }


def _card(c: I.Card) -> dict:
    return {"title": c.title, "takeaway": c.takeaway, "confidence": c.confidence, "kept": c.reliability,
            "n": c.games, "unit": c.unit}


def record_text(rec: dict) -> str:
    return f"{rec['wins']}–{rec['losses']}" if rec["games"] else "no games"


# --- printable report ---------------------------------------------------------------------
_CSS = """
body { font-family: "Segoe UI", Helvetica, Arial, sans-serif; color: #1f2933; max-width: 820px;
       margin: 24px auto; padding: 0 16px; line-height: 1.4; font-size: 13px; }
h1 { font-size: 22px; margin: 0 0 2px; color: #1d4e89; }
h2 { font-size: 15px; margin: 18px 0 6px; padding-bottom: 3px; border-bottom: 2px solid #d6e2f0; color: #1d4e89; }
.meta { color: #6b7785; font-size: 11px; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; }
table { border-collapse: collapse; width: 100%; }
td, th { text-align: left; padding: 3px 6px; border-bottom: 1px solid #eef2f6; vertical-align: top; }
th { font-size: 11px; color: #6b7785; font-weight: 600; }
.lead { font-weight: 600; }
.small { color: #6b7785; font-size: 11px; }
.badge { display: inline-block; padding: 0 6px; border-radius: 8px; background: #eef2f6; color: #4a5562; font-size: 11px; }
.warn { color: #b35c00; }
.proof { margin-top: 16px; padding: 8px 10px; background: #f3f7fb; border-left: 3px solid #2e6db4; }
footer { margin-top: 18px; color: #8a94a0; font-size: 10px; }
@media print { body { margin: 0 auto; } h2 { break-after: avoid; } tr, li { break-inside: avoid; } }
"""


def render_html(p: dict) -> str:
    """A one-page, text-only HTML version of Match prep (print it to PDF from a browser)."""
    e = html.escape
    form = lambda f: (f"{f['wins']}–{f['losses']} in {f['games']} games · last {f['last_n']}: "  # noqa: E731
                      f"{' '.join(f['last']) or '–'}")
    likely = "".join(
        f"<tr><td class='lead'>{e(x['champion'])}</td><td>{e(x['role'] or '')}</td><td>{x['prob']:.0%}</td>"
        f"<td class='small'>{e(x['evidence'])}{' · <span class=warn>small sample</span>' if x['small_sample'] else ''}"
        f"</td></tr>" for x in p["likely"])
    tells = "".join(f"<li>{e(t['takeaway'])} <span class='badge'>{t['n']} {t['unit']}</span></li>" for t in p["tells"])
    bans = "".join(
        f"<li><span class='lead'>{e(b['champion'])}</span>: {e(b['lead'])}"
        f"<div class='small'>{e(' · '.join(b['more']))}</div></li>" for b in p["bans"])
    beat = "".join(
        f"<li><span class='lead'>{e(c['title'])}.</span> {e(c['takeaway'])} <span class='badge'>{c['n']} {c['unit']}"
        f" · confidence {c['confidence']:.0%}</span></li>" for c in p["beat"])
    they_ban = "".join(
        f"<li>{e(x['champion'])}: banned in {x['bans']} of {x['games']} games"
        f"{' <span class=warn>(small sample)</span>' if x['small_sample'] else ''}</li>" for x in p["they_ban"])
    habits = "".join(f"<li>{e(h)}</li>" for h in p["habits"])
    h2h = p["h2h"]
    t = p["tabs"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Pre-match report: {e(p['our'])} vs {e(p['opp'])}</title>
<style>{_CSS}</style></head><body>
<h1>{e(p['our'])} vs {e(p['opp'])}: pre-match report</h1>
<div class="meta">Generated {e(p['generated'])} · data through {e(p['data_through'])} · records, form and bans: each
team's last {p['window']} patches · insights: last {p['insights_window']} patches</div>

<h2>Head-to-head and form</h2>
<table><tr><th>Head-to-head (window)</th><th>Head-to-head (all data)</th><th>{e(p['our'])}</th><th>{e(p['opp'])}</th></tr>
<tr><td>{record_text(h2h['window'])} <span class="badge">{h2h['window']['games']} games</span></td>
<td>{record_text(h2h['all'])} <span class="badge">{h2h['all']['games']} games</span></td>
<td>{e(form(p['form']['us']))}</td><td>{e(form(p['form']['them']))}</td></tr></table>

<div class="grid"><div>
<h2>What they'll draft</h2>
<div class="small">Most likely first-phase picks (details: {e(t['draft'])} tab)</div>
<table><tr><th>Champion</th><th>Role</th><th>Chance</th><th>Evidence</th></tr>{likely}</table>
<div class="small" style="margin-top:6px">Phase tells (details: {e(t['tells'])} tab)</div>
<ul>{tells or '<li class=small>No tell passes the bar.</li>'}</ul>
</div><div>
<h2>Our ban plan</h2>
<div class="small">Top 5 from {p['ban_games']} {e(p['opp'])} games, strongest reason first (details: {e(t['bans'])} tab)</div>
<ol>{bans or '<li class=small>No champion qualifies.</li>'}</ol>
<div class="small">{e(p['ban_proof'])}</div>
</div></div>

<h2>How to beat them</h2>
<div class="small">Insight cards that passed the bar (details: {e(t['beat'])} tab)</div>
<ul>{beat or '<li class=small>No card passes the bar in this window.</li>'}</ul>

<div class="grid"><div>
<h2>Watch out: what they ban against us</h2>
<div class="small">All head-to-head games (details: {e(t['vs'])} tab)</div>
<ul>{they_ban or '<li class=small>No head-to-head games.</li>'}</ul>
</div><div>
<h2>Watch out: our predictable habits</h2>
<div class="small">Self-scouting (details: {e(t['habits'])} tab)</div>
<ul>{habits}</ul>
</div></div>

<div class="proof">{e(p['proof'])}</div>
<footer>Draft Tendency Analyzer · GRID official match data (Sky's The Limit 2026 hackathon). Text only, no game
artwork. Every rate shows its sample size; see docs/VALIDATION.md for how each part was tested.</footer>
</body></html>"""
