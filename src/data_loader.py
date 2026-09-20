import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

SEASONS = {
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
    "2526": "2025/26",
}


def download_seasons():
    for code, season in SEASONS.items():
        url = BASE_URL.format(code)
        df = pd.read_csv(url)

        df = df.assign(Season=season)

        path = f"data/raw/{code}.csv"
        df.to_csv(path, index=False)

        print(f"{season}: {len(df)} matches -> {path}")
