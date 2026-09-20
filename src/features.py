"""Single-pass feature builder for match outcome prediction.

Walks through matches in date order, keeping running state per team.
Features for a match are computed BEFORE that match updates the state,
so there is no leakage.
"""

from collections import deque

import numpy as np
import pandas as pd

# ---- settings you can tune -------------------------------------------------
SHORT = 5            # short rolling window (matches)
LONG = 10            # long rolling window (matches)
MIN_MATCHES = 5      # skip a match if either team has fewer prior matches
START_ELO = 1500
K = 20               # Elo update speed
HOME_ADV = 60        # Elo points of home advantage
SEASON_REGRESSION = 0.2   # pull Elo 20% back to the mean each new season
# -----------------------------------------------------------------------------

# tuple layout stored per match: (points, goals_for, goals_against, sot_for, sot_against)
PTS, GF, GA, SOTF, SOTA = range(5)


class TeamState:
    def __init__(self):
        self.elo = START_ELO
        self.all = deque(maxlen=LONG)      # any venue
        self.home = deque(maxlen=SHORT)    # home matches only
        self.away = deque(maxlen=SHORT)    # away matches only


def _mean(records, idx, n=None):
    items = list(records)[-n:] if n else list(records)
    if not items:
        return np.nan
    return float(np.mean([r[idx] for r in items]))


def _gd(records, n=None):
    return _mean(records, GF, n) - _mean(records, GA, n)


def _sot_diff(records, n=None):
    return _mean(records, SOTF, n) - _mean(records, SOTA, n)


def _points(gf, ga):
    if gf > ga:
        return 3
    if gf == ga:
        return 1
    return 0


def _expected(elo_home, elo_away):
    return 1 / (1 + 10 ** ((elo_away - (elo_home + HOME_ADV)) / 400))


def _odds_probs(row):
    """Normalised bookmaker probabilities (removes the overround). None if missing."""
    try:
        h, d, a = row["B365H"], row["B365D"], row["B365A"]
    except KeyError:
        return None
    if pd.isna(h) or pd.isna(d) or pd.isna(a):
        return None
    inv = np.array([1 / h, 1 / d, 1 / a])
    inv = inv / inv.sum()
    return inv


def build_features(df, use_odds=False):
    """Return a DataFrame with one row per usable match.

    df needs: Date, Season, HomeTeam, AwayTeam, FTHG, FTAG, FTR.
    HST / AST (shots on target) are used if present, else treated as 0.
    Set use_odds=True to add Bet365 implied probabilities (needs B365H/D/A).
    """
    df = df.sort_values("Date", kind="stable").reset_index(drop=True)

    teams = {}
    rows = []
    current_season = None

    for row in df.itertuples(index=False):
        r = row._asdict()
        season = r["Season"]

        # new season: regress every Elo towards the mean
        if season != current_season:
            if current_season is not None:
                for t in teams.values():
                    t.elo = START_ELO + (1 - SEASON_REGRESSION) * (t.elo - START_ELO)
            current_season = season

        home, away = r["HomeTeam"], r["AwayTeam"]
        h = teams.setdefault(home, TeamState())
        a = teams.setdefault(away, TeamState())

        # ---------- features (computed before the update) ----------
        if len(h.all) >= MIN_MATCHES and len(a.all) >= MIN_MATCHES:
            feat = {
                "Date": r["Date"],
                "Season": season,
                "HomeTeam": home,
                "AwayTeam": away,
                "Elo_Diff": h.elo - a.elo,
                "Form5_Diff": _mean(h.all, PTS, SHORT) - _mean(a.all, PTS, SHORT),
                "Form10_Diff": _mean(h.all, PTS, LONG) - _mean(a.all, PTS, LONG),
                "GD10_Diff": _gd(h.all, LONG) - _gd(a.all, LONG),
                "SoT10_Diff": _sot_diff(h.all, LONG) - _sot_diff(a.all, LONG),
                # home team's form at home vs away team's form away
                "Venue_Form_Diff": _mean(h.home, PTS) - _mean(a.away, PTS),
                "Target": r["FTR"],
            }

            if use_odds:
                p = _odds_probs(r)
                if p is None:
                    feat = None
                else:
                    feat["Odds_H"], feat["Odds_D"], feat["Odds_A"] = p

            if feat is not None:
                rows.append(feat)

        # ---------- update state with the actual result ----------
        fthg, ftag = r["FTHG"], r["FTAG"]
        hst = r.get("HST", 0) if isinstance(r.get("HST", 0), (int, float)) else 0
        ast = r.get("AST", 0) if isinstance(r.get("AST", 0), (int, float)) else 0
        hst = 0 if pd.isna(hst) else hst
        ast = 0 if pd.isna(ast) else ast

        h_rec = (_points(fthg, ftag), fthg, ftag, hst, ast)
        a_rec = (_points(ftag, fthg), ftag, fthg, ast, hst)

        h.all.append(h_rec)
        h.home.append(h_rec)
        a.all.append(a_rec)
        a.away.append(a_rec)

        score_h = {"H": 1.0, "D": 0.5, "A": 0.0}[r["FTR"]]
        delta = K * (score_h - _expected(h.elo, a.elo))
        h.elo += delta
        a.elo -= delta

    out = pd.DataFrame(rows)
    # a team with 5+ matches can still lack home/away-only history early on
    out["Venue_Form_Diff"] = out["Venue_Form_Diff"].fillna(0.0)
    return out