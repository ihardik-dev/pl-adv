import pandas as pd
from pathlib import Path


def combine_seasons():
    files = Path("data/raw").glob("*.csv")

    dataframes = []

    for file in files:
        df = pd.read_csv(file)
        dataframes.append(df)

    combined = pd.concat(dataframes, ignore_index=True)

    combined["Date"] = pd.to_datetime(
        combined["Date"],
        dayfirst=True
    )

    combined = combined.sort_values("Date")

    output = "data/processed/matches.csv"
    combined.to_csv(output, index=False)

    print(f"Combined matches: {len(combined)}")
    print(f"Saved to: {output}")