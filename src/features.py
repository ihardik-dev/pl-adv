import pandas as pd


def get_team_result(row, team):
    if row["HomeTeam"] == team:
        goals_for = row["FTHG"]
        goals_against = row["FTAG"]
    else:
        goals_for = row["FTAG"]
        goals_against = row["FTHG"]

    if goals_for > goals_against:
        return 3
    elif goals_for == goals_against:
        return 1
    else:
        return 0


def get_previous_matches(df, team, date):
    previous = df[
        (
            (df["HomeTeam"] == team) |
            (df["AwayTeam"] == team)
        )
        & (df["Date"] < date)
    ]

    return previous


def get_last_5_form(df, team, date):
    previous = get_previous_matches(df, team, date)

    last_5 = previous.tail(5)

    if len(last_5) < 5:
        return None

    points = [
        get_team_result(row, team)
        for _, row in last_5.iterrows()
    ]

    return sum(points) / 5


def create_match_features(row, df):
    home_form = get_last_5_form(
        df,
        row["HomeTeam"],
        row["Date"]
    )

    away_form = get_last_5_form(
        df,
        row["AwayTeam"],
        row["Date"]
    )

    if home_form is None or away_form is None:
        return None

    return {
    "Date": row["Date"],
    "Season": row["Season"],
    "HomeTeam": row["HomeTeam"],
    "AwayTeam": row["AwayTeam"],

    "Home_Form": home_form,
    "Away_Form": away_form,

    "Target": row["FTR"]
    }