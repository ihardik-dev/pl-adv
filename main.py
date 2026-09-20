import pandas as pd
from src.features import create_match_features
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


df = pd.read_csv("data/processed/matches.csv")
df["Date"] = pd.to_datetime(df["Date"])

features = []

for _, row in df.iterrows():
    result = create_match_features(row, df)

    if result is not None:
        features.append(result)

features_df = pd.DataFrame(features)

features_df.to_csv(
    "data/processed/features.csv",
    index=False
)

features_df = pd.read_csv("data/processed/features.csv")
features_df["Date"] = pd.to_datetime(features_df["Date"])

train = features_df[
    features_df["Season"] != "2025/26"
]

# Test data
test = features_df[
    features_df["Season"] == "2025/26"
]

# Features used by the model
feature_columns = [
    "Home_Form",
    "Away_Form"
]

X_train = train[feature_columns]
X_test = test[feature_columns]

y_train = train["Target"]
y_test = test["Target"]

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["H", "D", "A"]
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["H", "D", "A"],
        zero_division=0
    )
)

