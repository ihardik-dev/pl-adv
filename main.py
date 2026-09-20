import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.features import build_features

USE_ODDS = False   # set True to add Bet365 implied probabilities
LABELS = ["H", "D", "A"]

# ---------------------------------------------------------------- data
df = pd.read_csv("data/processed/matches.csv")
df["Date"] = pd.to_datetime(df["Date"])

features_df = build_features(df, use_odds=USE_ODDS)
features_df.to_csv("data/processed/features.csv", index=False)

train = features_df[features_df["Season"] != "2025/26"]
test = features_df[features_df["Season"] == "2025/26"]

feature_columns = [
    "Elo_Diff",
    "Form5_Diff",
    "Form10_Diff",
    "GD10_Diff",
    "SoT10_Diff",
    "Venue_Form_Diff",
]
if USE_ODDS:
    feature_columns += ["Odds_H", "Odds_D", "Odds_A"]

X_train, y_train = train[feature_columns], train["Target"]
X_test, y_test = test[feature_columns], test["Target"]

print(f"Train matches: {len(train)}  |  Test matches: {len(test)}")

# ---------------------------------------------------------------- baseline
home_rate = (y_test == "H").mean()
print(f"\nBaseline (always predict Home): {home_rate:.3f}")

# ---------------------------------------------------------------- models
models = {
    "LogisticRegression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000)
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.05, max_iter=150, random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, proba, labels=model.classes_)

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.3f}   Log loss: {ll:.3f}")

    print("\nConfusion matrix (rows = actual, cols = predicted; H, D, A):")
    print(confusion_matrix(y_test, y_pred, labels=LABELS))

    print("\nClassification report:")
    print(
        classification_report(
            y_test, y_pred, labels=LABELS, zero_division=0
        )
    )