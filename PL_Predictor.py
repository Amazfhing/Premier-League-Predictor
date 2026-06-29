import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)

# import optuna
from xgboost import XGBClassifier
# from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt
import numpy as np

matches["date"] = pd.to_datetime(matches["date"])

odds = pd.read_csv("cleaned_odds.csv")
odds["Date"] = pd.to_datetime(odds["Date"])
# Map the mismatched names to match FBRef matches.csv naming
odds_mapping = {
    "Brighton": "Brighton and Hove Albion",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Newcastle": "Newcastle United",
    "Norwich": "Norwich City",
    "Tottenham": "Tottenham Hotspur",
    "West Brom": "West Bromwich Albion",
    "West Ham": "West Ham United",
    "Wolves": "Wolverhampton Wanderers"
}
odds["HomeTeam"] = odds["HomeTeam"].replace(odds_mapping)
odds["AwayTeam"] = odds["AwayTeam"].replace(odds_mapping)

# Reformat odds to be given from the perspective of the Team
home_odds = odds[["Date", "HomeTeam", "B365H", "B365D", "B365A"]].rename(
    columns={"Date": "date", "HomeTeam": "team", "B365H": "team_odds", "B365D": "draw_odds", "B365A": "opp_odds"}
)
away_odds = odds[["Date", "AwayTeam", "B365A", "B365D", "B365H"]].rename(
    columns={"Date": "date", "AwayTeam": "team", "B365A": "team_odds", "B365D": "draw_odds", "B365H": "opp_odds"}
)
team_odds = pd.concat([home_odds, away_odds])

matches = matches.merge(team_odds, on=["date", "team"], how="left")

matches = matches.dropna(subset=["team_odds", "draw_odds", "opp_odds"])

result_map = {"L": 0, "D": 1, "W": 2}
matches["target"] = matches["result"].map(result_map)

# Add Points column (Win=3, Draw=1, Loss=0)
matches["points"] = matches["target"].map({2: 3, 1: 1, 0: 0})

# One-Hot Encoding for categorical features
matches = pd.concat([matches, pd.get_dummies(matches[["venue", "opponent"]], drop_first=True)], axis=1)

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(
    "int")  ## converting hours to number in case a team plays better at a certain time
matches["day"] = matches["date"].dt.dayofweek  ## converting day of week of game to a number

# Generate a list of predictors dynamically based on one-hot encoding
predictors = ["hour", "day"] + [col for col in matches.columns if
                                col.startswith("venue_") or col.startswith("opponent_")]

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester United").sort_values("date")


def rolling_averages(group, cols, new_cols):  # function to take into consideration form of a team
    group = group.sort_values("date")
    
    # 1. Rest Days (days since last match)
    group["rest_days"] = group["date"].diff().dt.days
    
    # 2. Short-term form (3-game rolling average)
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    
    # 3. Long-term form (10-game rolling average)
    new_cols_10 = [f"{c}_10_rolling" for c in cols]
    group[new_cols_10] = group[cols].rolling(10, closed='left').mean()
    
    # 4. Rolling Points (sum of points over last 5 games)
    group["points_last_5"] = group["points"].rolling(5, closed='left').sum()
    

    group = group.dropna(subset=new_cols + new_cols_10 + ["rest_days", "points_last_5"])
    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]

if "xg" in matches.columns and "xga" in matches.columns:
    cols.extend(["xg", "xga"])

new_cols = [f"{c}_rolling" for c in cols]  # creating new columns with rolling average values
new_cols_10 = [f"{c}_10_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

matches_rolling.index = range(matches_rolling.shape[0])

features = predictors + new_cols + new_cols_10 + ["rest_days", "points_last_5"]

if "team_odds" in matches_rolling.columns:
    features.extend(["team_odds", "draw_odds", "opp_odds"])
# Uncomment the block of code below when we want to re-tune the hyper-parameters again
# def objective(trial):
#     param = {
#         "n_estimators": trial.suggest_int("n_estimators", 50, 200),
#         "max_depth": trial.suggest_int("max_depth", 3, 10),
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "random_state": 1,
#         "objective": "multi:softmax",
#         "num_class": 3
#     }
#     xgb = XGBClassifier(**param) #Feeds the parameters into the actual XGBoost model
#
#     tscv = TimeSeriesSplit(n_splits=3)
#     precisions = []
#
#     data = matches_rolling[matches_rolling["date"] < '2022-01-01'].sort_values("date")
#     X = data[features]
#     y = data["target"]
#
#     for train_index, test_index in tscv.split(X):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
#
#         xgb.fit(X_train, y_train)
#         preds = xgb.predict(X_test)
#         precisions.append(precision_score(y_test, preds, average="macro", zero_division=0))
#
#     return sum(precisions) / len(precisions)

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=200)
# best_params = study.best_params
# best_params["random_state"] = 1
# best_params["objective"] = "multi:softmax"
# best_params["num_class"] = 3

# Best parameters gathered from previous Optuna runs
best_params = {
    "n_estimators": 179,
    "max_depth": 3,
    "learning_rate": 0.010396244389156468,
    "subsample": 0.6546497502252517,
    "colsample_bytree": 0.633452310765801,
    "random_state": 1,
    "objective": "multi:softmax",
    "num_class": 3
}

model = XGBClassifier(**best_params)

def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] >= '2022-01-01']
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds, average="macro", zero_division=0)
    return combined, precision


combined, precision = make_predictions(matches_rolling, features)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


class MissingDict(dict):
    __missing__ = lambda self, key: key  # case when a team name is missing


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
accuracy = accuracy_score(combined["actual"], combined["prediction"])

print("\n Model Evaluation: ")
print(f"Accuracy:  {accuracy:.2%} | Precision: {precision:.2%}")
print("\nConfusion Matrix (0=L, 1=D, 2=W):")

print(pd.crosstab(
    index=combined["actual"],
    columns=combined["prediction"],
    rownames=["Actual"],
    colnames=["Pred"]
))

# Evaluate "Strong" Predictions
strong_preds = merged[(merged["prediction_x"] == 2) & (merged["prediction_y"] == 0)]

strong_draws = merged[(merged["prediction_x"] == 1) & (merged["prediction_y"] == 1)]

if not strong_preds.empty:
    strong_accuracy = accuracy_score(strong_preds["actual_x"], strong_preds["prediction_x"])
    print(f"\nStrong Predictions (Both Models Agree): {len(strong_preds)} matches | Accuracy: {strong_accuracy:.2%}")
else:
    print("\nStrong Predictions: 0 matches found")

if not strong_draws.empty:
    print(f"Strong Draws Accuracy: {accuracy_score(strong_draws['actual_x'], strong_draws['prediction_x']):.2%}")

# --- Feature Importance Chart ---
importances = model.feature_importances_
indices = np.argsort(importances)[::-1][:20]  # Get top 20 directly

plt.figure(figsize=(10, 8))
plt.title("Top 20 Feature Importances (XGBoost)")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
