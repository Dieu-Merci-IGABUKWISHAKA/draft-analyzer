"""Insight cards: data-backed counter-strategies from tendencies and records.

The win model adds no reliable accuracy beyond team strength (docs/MODEL.md), so
these cards don't use it. They compare a team's own records across situations
(champion open vs banned, comfort picks used up in a Fearless series, top champions
drafted or not, side, first pick) and describe their draft habits (phase tells,
predictability). Every rate has a Wilson 90% interval and its game count.

A card is shown only if every group it compares has at least MIN_GROUP games.
Win-effect cards are ranked by score = |effect| x confidence, where confidence =
1 - p-value of a two-proportion z-test. Many cards are tested per team, so some
gaps will be chance: `replication` checks how often gaps found in earlier games
point the same way in later games. Gaps are associations, not proof of cause.

Thresholds are fixed in advance (not tuned on results).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):  # allow running this file directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from draft_analyzer import model as M  # noqa: E402
from draft_analyzer.analysis import PHASE1_PICKS, _patch_key, fearless_game_ids  # noqa: E402

Z90 = 1.6449  # two-sided 90%
MIN_GROUP = 8  # games per compared group for a win-effect card
MIN_PICKS = 5  # a champion is in a team's pool with this many picks in the window
TOP_COMFORT = 5  # "top champions" = their 5 most-picked
MIN_TELL_PICKS = 6  # picks needed for a phase tell
PHASE1_BASE = 0.6  # 6 of a draft's 10 picks are in phase 1
TELL_HIGH, TELL_LOW = 0.85, 0.30  # "always early" / "held for counter-pick" phase-1 shares
MIN_PRED_GAMES = 15  # games needed to compare a team's predictability with the league
DEFAULT_PATCHES = 12
# A card reaches "How to beat" only if it is at least MIN_CONFIDENCE confident, and its kind
# of pattern kept at least MIN_RELIABILITY of its gap in later games (counting only that
# kind's confident cards; see replication_summary). Both were set after seeing the
# replication results (docs/INSIGHTS.md).
MIN_RELIABILITY = 0.25
MIN_CONFIDENCE = 0.80


def holds_up(kept: float, late_gap_lo: float) -> bool:
    """A kind of pattern holds up if, among its confident cases, at least MIN_RELIABILITY of
    the gap survived in later games AND the 90% interval of the later gap is above zero.
    (The second condition was added in the macro phase; it changes nothing here: every kind
    that kept >= 25% here also has a later-gap interval above zero.)"""
    return bool(kept >= MIN_RELIABILITY and late_gap_lo > 0)


# --- statistics ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = Z90) -> tuple[float, float]:
    """Wilson score interval for k successes in n trials."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def two_prop(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """(rate1 - rate2, confidence = 1 - two-sided p-value), pooled two-proportion z-test."""
    if min(n1, n2) == 0:
        return float("nan"), 0.0
    p1, p2 = k1 / n1, k2 / n2
    pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return p1 - p2, 0.0
    return p1 - p2, 1 - math.erfc(abs(p1 - p2) / se / math.sqrt(2))


def one_prop(k: int, n: int, p0: float) -> float:
    """Confidence (1 - two-sided p-value) that a rate differs from p0 (normal approximation)."""
    if n == 0:
        return 0.0
    se = math.sqrt(p0 * (1 - p0) / n)
    return 1 - math.erfc(abs(k / n - p0) / se / math.sqrt(2))


def entropy_bits(counts: Iterable[float]) -> float:
    """Miller-Madow corrected entropy (bits) of a count vector, capped at log2(distinct
    values seen) so the effective number of choices never exceeds what was observed."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    n = c.sum()
    if n == 0 or len(c) == 1:
        return 0.0
    p = c / n
    h = -(p * np.log2(p)).sum() + (len(c) - 1) / (2 * n * math.log(2))
    return float(min(h, math.log2(len(c))))


# --- cards --------------------------------------------------------------------------------
@dataclass
class Group:
    label: str
    wins: int
    games: int

    @property
    def rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    @property
    def ci(self) -> tuple[float, float]:
        return wilson(self.wins, self.games)

    def text(self) -> str:
        lo, hi = self.ci
        return (
            f"{self.label}: {self.rate:.0%} won ({self.wins}–{self.games - self.wins} in {self.games} games; "
            f"90% range {lo:.0%}–{hi:.0%})"
        )


@dataclass
class Card:
    kind: str  # ban impact / fearless carry-over / comfort dependency / side / first pick
    title: str  # the advice, short
    takeaway: str  # one sentence
    groups: list[Group]  # [better-for-them group, worse-for-them group]
    effect: float  # win-rate gap between the groups (>= 0)
    confidence: float  # 1 - p-value
    evidence: list[str] = field(default_factory=list)
    champions: list[str] = field(default_factory=list)
    reliability: float = 1.0  # share of this card type's gap that held up in later games
    typical: float = 1.0  # typical gap for this card type (so different units compare fairly)
    n: int | None = None  # sample size when there are no groups (phase tells: picks)
    unit: str = "games"

    @property
    def score(self) -> float:
        """Gap relative to the typical gap of its type x confidence x share that held up later."""
        return self.effect / self.typical * self.confidence * self.reliability

    @property
    def games(self) -> int:
        return self.n if self.n is not None else sum(g.games for g in self.groups)

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "title": self.title,
            "takeaway": self.takeaway,
            "effect": self.effect,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "score": self.score,
            "games": self.games,
            "unit": self.unit,
            "evidence": [g.text() for g in self.groups] + self.evidence,
        }


@dataclass
class Tell:
    champion: str
    picks: int
    phase1: int
    kind: str  # "early" or "held"
    takeaway: str
    confidence: float

    @property
    def share(self) -> float:
        return self.phase1 / self.picks

    @property
    def score(self) -> float:
        return abs(self.share - PHASE1_BASE) * self.confidence

    def as_card(self) -> Card:
        lo, hi = wilson(self.phase1, self.picks)
        title = (
            f"{self.champion} comes out early"
            if self.kind == "early"
            else f"{self.champion} is held for the counter-pick"
        )
        return Card(
            "phase tell",
            title,
            self.takeaway,
            [],
            abs(self.share - PHASE1_BASE),
            self.confidence,
            [
                f"Picked in phase 1 in {self.phase1} of {self.picks} picks: {self.share:.0%} "
                f"(90% range {lo:.0%}–{hi:.0%}); a typical champion: {PHASE1_BASE:.0%}."
            ],
            [self.champion],
            n=self.picks,
            unit="picks",
        )


def _compare(
    kind: str,
    high: Group,
    low: Group,
    title: str,
    takeaway: str,
    evidence: Iterable[str] = (),
    champions: Iterable[str] = (),
    min_group: int = MIN_GROUP,
) -> Card | None:
    """A card comparing two groups, oriented so `high` is the better group for the team;
    None unless both groups reach `min_group` games."""
    if min(high.games, low.games) < min_group:
        return None
    diff, conf = two_prop(high.wins, high.games, low.wins, low.games)
    if diff < 0:
        return None
    return Card(kind, title, takeaway, [high, low], diff, conf, list(evidence), list(champions))


# --- the scouting engine ------------------------------------------------------------------
class Scout:
    """Builds insight cards for any team. Construct once (it prepares the tables)."""

    def __init__(self, drafts: pd.DataFrame, min_group: int = MIN_GROUP, min_tell_picks: int = MIN_TELL_PICKS):
        self.drafts = drafts
        self.min_group, self.min_tell_picks = min_group, min_tell_picks
        self.games, self.actions = M.prepare(drafts)
        a = self.actions.join(self.games[["series_id", "game_number"]], on="game_id")
        self.actions = a
        self.first_use = a[a["action"] == "pick"].groupby(["series_id", "champion_id"])["game_number"].min()
        self.fearless = fearless_game_ids(drafts)
        self.champ_names = drafts.drop_duplicates("champion_id").set_index("champion_id")["champion_name"].astype(str)
        self.team_names = drafts.sort_values("date").groupby("team_id")["team_name"].last().astype(str)

    # --- tables ---------------------------------------------------------------------------
    def team_games(
        self,
        team: str,
        last_patches: int | None = DEFAULT_PATCHES,
        before: str | pd.Timestamp | None = None,
        since: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """The team's games (index game_id): won, side, first_pick, series_id, game_number,
        opponent. Window: its last N patches with games, optionally within [since, before)."""
        g = self.games[(self.games["blue_team"] == team) | (self.games["red_team"] == team)]
        if before is not None:
            g = g[g["date"] < pd.Timestamp(before)]
        if since is not None:
            g = g[g["date"] >= pd.Timestamp(since)]
        if last_patches is not None and len(g):
            keep = sorted(g["patch"].dropna().unique(), key=_patch_key)[-last_patches:]
            g = g[g["patch"].isin(keep)]
        blue = g["blue_team"] == team
        return pd.DataFrame(
            {
                "won": np.where(blue, g["blue_won"], 1 - g["blue_won"]).astype(int),
                "side": np.where(blue, "blue", "red"),
                "first_pick": np.where(blue, g["blue_first_pick"], 1 - g["blue_first_pick"]).astype(bool),
                "series_id": g["series_id"],
                "game_number": g["game_number"].astype(int),
                "opponent": np.where(blue, g["red_team"], g["blue_team"]),
                "date": g["date"],
            },
            index=g.index,
        )

    def _picks(self, team: str, tg: pd.DataFrame) -> pd.DataFrame:
        a = self.actions
        return a[(a["team_id"] == team) & (a["action"] == "pick") & a["game_id"].isin(tg.index)]

    def pool(self, team: str, tg: pd.DataFrame) -> pd.Series:
        """champion_id -> picks in the window, most picked first."""
        return self._picks(team, tg)["champion_id"].value_counts()

    def _locked(self, tg: pd.DataFrame, champ: str) -> np.ndarray:
        """Games in which `champ` was already used earlier in the series (hard Fearless)."""
        fu = pd.MultiIndex.from_arrays([tg["series_id"], np.full(len(tg), champ)])
        first = self.first_use.reindex(fu).to_numpy()
        return (tg.index.isin(self.fearless)) & ~np.isnan(first) & (first < tg["game_number"].to_numpy())

    # --- wording --------------------------------------------------------------------------
    def _who(self, team: str, view: str) -> str:
        return "we" if view == "self" else self.team_names.get(team, team)

    @staticmethod
    def _cap(s: str) -> str:
        return s[:1].upper() + s[1:]

    def _name(self, c: str) -> str:
        return self.champ_names.get(c, c)

    # --- card types -----------------------------------------------------------------------
    def ban_impact(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Card]:
        """Their record when a pool champion is open vs banned (Fearless-locked games left out)."""
        who, cards = self._who(team, view), []
        bans = self.actions[(self.actions["action"] == "ban") & self.actions["game_id"].isin(tg.index)]
        banned_by = bans.groupby("champion_id")["game_id"].apply(set)
        pool = self.pool(team, tg)
        for c, picks in pool[pool >= MIN_PICKS].items():
            banned = tg.index.isin(banned_by.get(c, set()))
            locked = self._locked(tg, c)
            open_ = ~banned & ~locked
            g_open = Group(f"{self._name(c)} open", int(tg.loc[open_, "won"].sum()), int(open_.sum()))
            g_ban = Group(f"{self._name(c)} banned", int(tg.loc[banned, "won"].sum()), int(banned.sum()))
            name = self._name(c)
            extra = [f"{self._cap(who)} picked it in {picks} of {len(tg)} games in the window."]
            if g_open.rate >= g_ban.rate:
                if view == "self":
                    title, take = (
                        f"Opponents can ban {name}",
                        f"We win {g_open.rate:.0%} when {name} is open and {g_ban.rate:.0%} when it's banned: "
                        f"expect it banned and have a fallback ready.",
                    )
                else:
                    title, take = (
                        f"Ban {name}",
                        f"{who} win {g_open.rate:.0%} when {name} is open and {g_ban.rate:.0%} when it's "
                        f"banned: banning it takes away a real edge.",
                    )
                card = _compare("ban impact", g_open, g_ban, title, take, extra, [name], min_group=self.min_group)
            else:
                if view == "self":
                    title, take = (
                        f"Bans on {name} don't hurt us",
                        f"We win {g_ban.rate:.0%} when {name} is banned vs {g_open.rate:.0%} when open: "
                        f"it isn't the crutch opponents may think.",
                    )
                else:
                    title, take = (
                        f"Don't spend a ban on {name}",
                        f"{who} win more when {name} is banned ({g_ban.rate:.0%}) than when it's open "
                        f"({g_open.rate:.0%}): ban something else.",
                    )
                card = _compare("ban impact", g_ban, g_open, title, take, extra, [name], min_group=self.min_group)
            if card:
                cards.append(card)
        return cards

    def fearless_carryover(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Card]:
        """Later games (G2-G5) of Fearless series: record while a top champion is still
        available vs after it was used earlier in the series (by either team)."""
        who, cards = self._who(team, view), []
        later = tg[tg.index.isin(self.fearless) & (tg["game_number"] >= 2)]
        top = list(self.pool(team, tg).index[:TOP_COMFORT])
        if later.empty or not top:
            return cards
        gone = np.column_stack([self._locked(later, c) for c in top])
        for i, c in enumerate(top):
            name = self._name(c)
            g_av = Group(f"{name} still available", int(later.loc[~gone[:, i], "won"].sum()), int((~gone[:, i]).sum()))
            g_gone = Group(f"{name} already used", int(later.loc[gone[:, i], "won"].sum()), int(gone[:, i].sum()))
            if g_av.rate >= g_gone.rate:
                if view == "self":
                    title = f"Saving {name} matters in a series"
                    take = (
                        f"In games 2–5 we win {g_av.rate:.0%} while {name} is still available and "
                        f"{g_gone.rate:.0%} once it's been used: an opponent can burn it early."
                    )
                else:
                    title = f"Get {name} used up early in the series"
                    take = (
                        f"In games 2–5 {who} win {g_av.rate:.0%} while {name} is available and "
                        f"{g_gone.rate:.0%} once it's gone: in Fearless, pick it yourself early so they lose it."
                    )
                card = _compare(
                    "fearless carry-over", g_av, g_gone, title, take, champions=[name], min_group=self.min_group
                )
            else:
                card = _compare(
                    "fearless carry-over",
                    g_gone,
                    g_av,
                    f"{name} gone doesn't slow {'us' if view == 'self' else who} down",
                    f"In games 2–5 {who} win {g_gone.rate:.0%} after {name} is used vs "
                    f"{g_av.rate:.0%} while it's available: burning it isn't worth a plan.",
                    champions=[name],
                    min_group=self.min_group,
                )
            if card:
                cards.append(card)
        n_gone = gone.sum(axis=1)
        names = ", ".join(self._name(c) for c in top)
        g0 = Group("0–1 of top 5 used", int(later.loc[n_gone <= 1, "won"].sum()), int((n_gone <= 1).sum()))
        g2 = Group("2+ of top 5 used", int(later.loc[n_gone >= 2, "won"].sum()), int((n_gone >= 2).sum()))
        if g0.rate >= g2.rate:
            take = (
                f"In games 2–5 {who} win {g0.rate:.0%} with most of their top 5 ({names}) left and "
                f"{g2.rate:.0%} once 2+ are gone: burn their pool across the series."
                if view == "opponent"
                else f"In games 2–5 we win {g0.rate:.0%} with most of our top 5 left and {g2.rate:.0%} once 2+ "
                f"are gone: opponents can grind our pool down across a series."
            )
            card = _compare(
                "fearless carry-over",
                g0,
                g2,
                "Burn their pool across the series" if view == "opponent" else "Our pool runs thin late in series",
                take,
                [f"Top 5: {names}."],
                [self._name(c) for c in top],
                min_group=self.min_group,
            )
            if card:
                cards.append(card)
        return cards

    def comfort_dependency(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Card]:
        """Record with 2+ of their top 5 champions in the game vs 0-1, plus the share of
        their picks in wins that are top-5 champions."""
        who = self._who(team, view)
        picks = self._picks(team, tg)
        top = list(self.pool(team, tg).index[:TOP_COMFORT])
        if not top:
            return []
        k = picks[picks["champion_id"].isin(top)].groupby("game_id").size().reindex(tg.index).fillna(0)
        many, few = k >= 2, k <= 1
        g_many = Group("2+ of top 5 drafted", int(tg.loc[many, "won"].sum()), int(many.sum()))
        g_few = Group("0–1 of top 5 drafted", int(tg.loc[few, "won"].sum()), int(few.sum()))
        won = picks["game_id"].map(tg["won"]) == 1
        in_top = picks["champion_id"].isin(top)
        share_w = in_top[won].mean() if won.any() else float("nan")
        share_l = in_top[~won].mean() if (~won).any() else float("nan")
        names = ", ".join(self._name(c) for c in top)
        ev = [
            f"Top 5: {names}.",
            f"Top-5 champions are {share_w:.0%} of {'our' if view == 'self' else 'their'} picks in wins "
            f"and {share_l:.0%} in losses ({int(won.sum())} and {int((~won).sum())} picks).",
        ]
        if g_many.rate >= g_few.rate:
            if view == "self":
                title = "We lean on our top 5"
                take = (
                    f"We win {g_many.rate:.0%} with 2+ of our top 5 and {g_few.rate:.0%} with fewer: "
                    f"opponents who ban or take them pull us off our best drafts."
                )
            else:
                title = "Deny their top 5"
                take = (
                    f"{who} win {g_many.rate:.0%} with 2+ of {names} and {g_few.rate:.0%} with fewer: "
                    f"spend bans and early picks on that pool."
                )
            card = _compare(
                "comfort dependency",
                g_many,
                g_few,
                title,
                take,
                ev,
                [self._name(c) for c in top],
                min_group=self.min_group,
            )
        else:
            card = _compare(
                "comfort dependency",
                g_few,
                g_many,
                "Comfort picks aren't their engine" if view == "opponent" else "Our comfort picks aren't our engine",
                f"{self._cap(who)} win {g_few.rate:.0%} with 0–1 of the top 5 vs {g_many.rate:.0%} with 2+: "
                f"denying the pool won't slow {'us' if view == 'self' else 'them'} down.",
                ev,
                min_group=self.min_group,
            )
        return [card] if card else []

    def side_and_first_pick(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Card]:
        who, cards = self._who(team, view), []
        blue, red = tg["side"] == "blue", tg["side"] == "red"
        g_b = Group("on blue", int(tg.loc[blue, "won"].sum()), int(blue.sum()))
        g_r = Group("on red", int(tg.loc[red, "won"].sum()), int(red.sum()))
        pref = [f"{self._cap(who)} played {blue.mean():.0%} of games on blue."]
        hi, lo = (g_b, g_r) if g_b.rate >= g_r.rate else (g_r, g_b)
        weak = lo.label.split()[-1]
        title = f"Put them on {weak}" if view == "opponent" else f"We're weaker on {weak}"
        take = (
            f"{who} win {hi.rate:.0%} {hi.label} and {lo.rate:.0%} {lo.label}: with first selection, "
            f"take the other side so they play {weak}."
            if view == "opponent"
            else f"We win {hi.rate:.0%} {hi.label} and {lo.rate:.0%} {lo.label}: opponents with first selection "
            f"can leave us {weak}."
        )
        if card := _compare("side", hi, lo, title, take, pref, min_group=self.min_group):
            cards.append(card)

        fp, nf = tg["first_pick"], ~tg["first_pick"]
        g_fp = Group("with first pick", int(tg.loc[fp, "won"].sum()), int(fp.sum()))
        g_nf = Group("without first pick", int(tg.loc[nf, "won"].sum()), int(nf.sum()))
        hi, lo = (g_fp, g_nf) if g_fp.rate >= g_nf.rate else (g_nf, g_fp)
        if view == "opponent":
            title = "Take first pick from them" if hi is g_fp else "Let them have first pick"
            take = f"{who} win {hi.rate:.0%} {hi.label} and {lo.rate:.0%} {lo.label}: " + (
                "keep first pick for yourself."
                if hi is g_fp
                else "first pick doesn't help them; take the side instead."
            )
        else:
            title = "We need first pick" if hi is g_fp else "We're fine without first pick"
            take = f"We win {hi.rate:.0%} {hi.label} and {lo.rate:.0%} {lo.label}: " + (
                "opponents with first selection can take it from us."
                if hi is g_fp
                else "first selection can go to side instead."
            )
        if card := _compare(
            "first pick",
            hi,
            lo,
            title,
            take,
            [f"{self._cap(who)} had first pick in {fp.mean():.0%} of games."],
            min_group=self.min_group,
        ):
            cards.append(card)
        return cards

    def phase_tells(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Tell]:
        """Champions they nearly always take in the first pick phase, or hold for phase 2."""
        who = self._who(team, view)
        picks = self._picks(team, tg)
        stats = (
            picks.assign(p1=picks["draft_order"].isin(PHASE1_PICKS)).groupby("champion_id")["p1"].agg(["size", "sum"])
        )
        tells = []
        for c, r in stats[stats["size"] >= self.min_tell_picks].iterrows():
            n, k, name = int(r["size"]), int(r["sum"]), self._name(c)
            lo, hi = wilson(k, n)
            if k / n >= TELL_HIGH and lo > PHASE1_BASE:
                take = (
                    f"{who} take {name} early ({k} of {n} picks in phase 1): ban it in phase 1 or take it first; "
                    f"don't plan to answer it late."
                    if view == "opponent"
                    else f"We take {name} early ({k} of {n} in phase 1): opponents can plan a phase-1 ban."
                )
                tells.append(Tell(name, n, k, "early", take, one_prop(k, n, PHASE1_BASE)))
            elif k / n <= TELL_LOW and hi < PHASE1_BASE:
                take = (
                    f"{who} hold {name} for the counter-pick ({n - k} of {n} picks in phase 2): keep that lane's "
                    f"pick late or ban it in phase 2."
                    if view == "opponent"
                    else f"We hold {name} for the counter-pick ({n - k} of {n} in phase 2): "
                    "opponents can ban it in phase 2."
                )
                tells.append(Tell(name, n, k, "held", take, one_prop(k, n, PHASE1_BASE)))
        return sorted(tells, key=lambda t: -t.score)

    # --- predictability -------------------------------------------------------------------
    def first_choices(self, team: str, tg: pd.DataFrame, kind: str) -> pd.Series:
        """The team's first pick (or first ban) champion in each game."""
        a = self.actions
        a = a[(a["team_id"] == team) & (a["action"] == kind) & a["game_id"].isin(tg.index)]
        return a.sort_values("draft_order").drop_duplicates("game_id")["champion_id"]

    def predictability(self, team: str, last_patches: int | None = DEFAULT_PATCHES) -> dict:
        """How spread out the team's first picks and first bans are: entropy, effective
        number of choices (2^entropy), top-3 share, compared with the league."""
        out = {}
        league_teams = self.team_names.index
        for kind in ("pick", "ban"):
            own = self._spread(self.first_choices(team, self.team_games(team, last_patches), kind))
            league = [
                self._spread(self.first_choices(t, tg, kind))
                for t in league_teams
                if t != team and len(tg := self.team_games(t, last_patches)) >= MIN_PRED_GAMES
            ]
            top3 = [x["top3_share"] for x in league]
            own["league_median_top3"] = float(np.median(top3)) if top3 else float("nan")
            own["league_teams"] = len(top3)
            own["more_predictable_than"] = (
                float(np.mean([own["top3_share"] > t for t in top3])) if top3 else float("nan")
            )
            own["top"] = [self._name(c) for c in own.pop("top_ids")]
            out[kind] = own
        return out

    @staticmethod
    def _spread(s: pd.Series) -> dict:
        vc = s.value_counts()
        n = int(vc.sum())
        h = entropy_bits(vc.to_numpy())
        return {
            "games": n,
            "distinct": int(len(vc)),
            "entropy_bits": h,
            "effective_choices": 2**h,
            "top3_share": float(vc.iloc[:3].sum() / n) if n else float("nan"),
            "top_ids": list(vc.index[:3]),
            "top_counts": [int(x) for x in vc.iloc[:3]],
        }

    def predictability_takeaways(self, team: str, pred: dict, view: str = "self") -> list[str]:
        who = self._who(team, view)
        out = []
        for kind, label in (("pick", "first pick"), ("ban", "first ban")):
            p = pred[kind]
            if not p["games"]:
                continue
            tops = ", ".join(f"{c} ({k})" for c, k in zip(p["top"], p["top_counts"], strict=False))
            rank = p["more_predictable_than"]
            more = rank >= 0.5
            if view == "self":
                verdict = (
                    f"Opponents can prepare for it: plan bans or first picks around {p['top'][0]}."
                    if more
                    else "That's hard to prepare for."
                )
            else:
                verdict = f"Prepare for {p['top'][0]}." if more else "Hard to predict: prepare broadly."
            out.append(
                f"{label.capitalize()}: {who} {'are' if who == 'we' else 'is'} more predictable than "
                f"{rank:.0%} of {p['league_teams']} teams. Top 3 = {p['top3_share']:.0%} of {p['games']} games "
                f"({tops}); league median {p['league_median_top3']:.0%}. {verdict}"
            )
        return out

    # --- everything for one team ----------------------------------------------------------
    def cards_from(self, team: str, tg: pd.DataFrame, view: str = "opponent") -> list[Card]:
        return (
            self.ban_impact(team, tg, view)
            + self.fearless_carryover(team, tg, view)
            + self.comfort_dependency(team, tg, view)
            + self.side_and_first_pick(team, tg, view)
        )

    def cards(self, team: str, last_patches: int | None = DEFAULT_PATCHES, view: str = "opponent") -> list[Card]:
        return sorted(self.cards_from(team, self.team_games(team, last_patches), view), key=lambda c: -c.score)

    def report(
        self,
        team: str,
        last_patches: int | None = DEFAULT_PATCHES,
        view: str = "opponent",
        top: int = 5,
        replication: pd.DataFrame | None = None,
    ) -> dict:
        """All cards (win-effect cards and phase tells) ranked by
            (gap / typical gap of its type) x confidence x share of that type's gap that held up later,
        with the typical gap and the share from replication_summary (both 1 if None). Dividing
        by the typical gap puts win-rate gaps and pick-phase gaps on one scale."""
        tg = self.team_games(team, last_patches)
        cards = self.cards_from(team, tg, view) + [t.as_card() for t in self.phase_tells(team, tg, view)]
        rep = {} if replication is None or replication.empty else replication.set_index("kind").to_dict("index")
        holds = {k: holds_up(v["survives"], v.get("late_gap_lo", float("inf"))) for k, v in rep.items()}
        for c in cards:
            if replication is not None:
                c.reliability = float(rep.get(c.kind, {}).get("survives", 0.0))
                c.typical = float(rep.get(c.kind, {}).get("early_gap", 1.0)) or 1.0
        cards = sorted(cards, key=lambda c: (-c.score, -c.effect))
        pred = self.predictability(team, last_patches)
        passes = [
            c for c in cards if c.confidence >= MIN_CONFIDENCE and (replication is None or holds.get(c.kind, False))
        ]
        return {
            "team": self.team_names.get(team, team),
            "games": len(tg),
            "cards": cards,
            "top": passes[:top],
            "passing": len(passes),
            "predictability": pred,
            "predictability_takeaways": self.predictability_takeaways(team, pred, view),
        }


# --- do the gaps hold up in later games? --------------------------------------------------
REPLICATION_SPLITS = ("2025-07-01", "2026-04-01")
LATE_MIN_GROUP = 3  # later periods are short, so later groups need only 3 games (picks for tells)


def _key(card: Card) -> tuple:
    return card.kind, frozenset(g.label for g in card.groups)


def replication(scout: Scout, splits: Sequence[str] = REPLICATION_SPLITS) -> pd.DataFrame:
    """For every team and split date: build the cards (and phase tells) from all games
    before the split, then the same comparisons from games after it. One row per card
    that can be checked: its earlier gap, its later gap (signed: > 0 = same direction)."""
    late_scout = Scout.__new__(Scout)
    late_scout.__dict__.update(scout.__dict__)
    late_scout.min_group, late_scout.min_tell_picks = LATE_MIN_GROUP, LATE_MIN_GROUP
    rows = []
    for split in splits:
        for t in scout.team_names.index:
            early, late = scout.team_games(t, None, before=split), scout.team_games(t, None, since=split)
            if len(early) < 2 * scout.min_group or len(late) < 2 * LATE_MIN_GROUP:
                continue
            later = {_key(c): c for c in late_scout.cards_from(t, late)}
            for c in scout.cards_from(t, early):
                lc = later.get(_key(c))
                if lc is not None:
                    same = lc.groups[0].label == c.groups[0].label
                    rows.append(
                        {
                            "split": split,
                            "team": t,
                            "kind": c.kind,
                            "title": c.title,
                            "early_effect": c.effect,
                            "early_conf": c.confidence,
                            "late_effect": lc.effect if same else -lc.effect,
                            "late_games": lc.games,
                        }
                    )
            late_picks = late_scout._picks(t, late)
            for tell in scout.phase_tells(t, early):
                cid = scout.champ_names[scout.champ_names == tell.champion].index[0]
                lp = late_picks[late_picks["champion_id"] == cid]
                if len(lp) >= LATE_MIN_GROUP:
                    share = lp["draft_order"].isin(PHASE1_PICKS).mean()
                    sign = 1 if tell.kind == "early" else -1
                    rows.append(
                        {
                            "split": split,
                            "team": t,
                            "kind": "phase tell",
                            "title": tell.takeaway,
                            "early_effect": abs(tell.share - PHASE1_BASE),
                            "early_conf": tell.confidence,
                            "late_effect": sign * (share - PHASE1_BASE),
                            "late_games": len(lp),
                        }
                    )
    return pd.DataFrame(rows)


def replication_summary(rep: pd.DataFrame, min_confidence: float = MIN_CONFIDENCE) -> pd.DataFrame:
    """Per card type, over the cards that pass the confidence bar (the ones the tab would
    show): checks, how often the later gap points the same way, the average gap before and
    after (with a 90% interval), and the share of the gap that survives ("kept"). The
    same-direction rate over all cards of the type is kept for comparison."""
    if rep.empty:
        return pd.DataFrame()
    out = []
    for kind, g_all in rep.groupby("kind"):
        g = g_all[g_all["early_conf"] >= min_confidence]
        if g.empty:
            continue
        sd = g["late_effect"].std(ddof=1) if len(g) > 1 else float("nan")
        half = Z90 * sd / math.sqrt(len(g)) if len(g) > 1 else float("nan")
        late = float(g["late_effect"].mean())
        out.append(
            {
                "kind": kind,
                "checks": len(g),
                "same_direction": float((g["late_effect"] > 0).mean()),
                "early_gap": float(g["early_effect"].mean()),
                "late_gap": late,
                "late_gap_lo": late - half,
                "late_gap_hi": late + half,
                "all_checks": len(g_all),
                "all_same_direction": float((g_all["late_effect"] > 0).mean()),
            }
        )
    out = pd.DataFrame(out)
    out["survives"] = (out["late_gap"] / out["early_gap"]).clip(0, 1)
    return out.sort_values("survives", ascending=False).reset_index(drop=True)


def load_replication(scout: Scout, rebuild: bool = False) -> pd.DataFrame:
    """replication_summary, cached in data_cache/ (aggregate numbers per card type only)."""
    import joblib

    from draft_analyzer.config import CACHE_DIR

    path = CACHE_DIR / "insights_replication.joblib"
    key = (
        str(scout.games["date"].max().date()),
        scout.min_group,
        scout.min_tell_picks,
        REPLICATION_SPLITS,
        LATE_MIN_GROUP,
        MIN_PICKS,
        TOP_COMFORT,
        MIN_CONFIDENCE,
        "confident-subset",
    )
    if not rebuild and path.exists():
        saved = joblib.load(path)
        if saved.get("key") == key:
            return saved["summary"]
    summary = replication_summary(replication(scout))
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"key": key, "summary": summary}, path)
    return summary


if __name__ == "__main__":
    from draft_analyzer.drafts import load_drafts

    pd.set_option("display.width", 200)
    sc = Scout(load_drafts())
    summary = load_replication(sc, rebuild=True)
    print(summary.round(3).to_string(index=False))
    for name, view in (("LYON", "opponent"), ("FlyQuest", "opponent"), ("Cloud9 Kia", "self")):
        r = sc.report(M._resolve_team(name, sc.drafts), view=view, replication=summary)
        print({k: sum(c.kind == k for c in r["cards"]) for k in summary["kind"]})
        print()
        print(f"== {name}: {r['games']} games, {len(r['cards'])} cards")
        for c in r["top"]:
            print(
                f"  [{c.kind}] {c.title}: effect {c.effect:.0%} conf {c.confidence:.2f} held up {c.reliability:.0%} "
                f"score {c.score:.3f} n={c.games} {c.unit}"
            )
            print(f"     {c.takeaway}")
        for t in r["predictability_takeaways"]:
            print("  pred:", t)
