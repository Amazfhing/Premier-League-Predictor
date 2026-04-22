## PL Predictor using scikit-learn to predict from the matches.csv stat sheet containing data from all matches from 2022-2020
import pandas as pd

matches = pd.read_csv("matches.csv", index_col=0)

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

##converting all objects to int or float to be processed by the machine learning software
matches["date"] = pd.to_datetime(matches["date"])

## Recommendation 5: Multi-Class Target (Win:2, Draw:1, Loss:0)
result_map = {"L": 0, "D": 1, "W": 2}
matches["target"] = matches["result"].map(result_map)

## Add Points column (Win=3, Draw=1, Loss=0)
matches["points"] = matches["target"].map({2: 3, 1: 1, 0: 0})

## Recommendation 2: One-Hot Encoding for categorical features
matches = pd.concat([matches, pd.get_dummies(matches[["venue", "opponent"]], drop_first=True)], axis=1)

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype(
    "int")  ## converting hours to number in case a team plays better at a certain time
matches["day"] = matches["date"].dt.dayofweek  ## converting day of week of game to a number

# Generate a list of predictors dynamically based on one-hot encoding
predictors = ["hour", "day"] + [col for col in matches.columns if
                                col.startswith("venue_") or col.startswith("opponent_")]

grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester United").sort_values("date")


def rolling_averages(group, cols, new_cols):  ## function to take into consideration form of a team
    group = group.sort_values("date")  ## sorting games by date
    
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
    
    # Drop missing values ensuring we have all features for the model
    group = group.dropna(subset=new_cols + new_cols_10 + ["rest_days", "points_last_5"])
    return group


cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
# Check if xg and xga exist in the dataset (Recommendation 1)
if "xg" in matches.columns and "xga" in matches.columns:
    cols.extend(["xg", "xga"])

new_cols = [f"{c}_rolling" for c in cols]  ## creating new columns with rolling average values
new_cols_10 = [f"{c}_10_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')  ## dropping extra index level

matches_rolling.index = range(matches_rolling.shape[0])  ## adding new index

features = predictors + new_cols + new_cols_10 + ["rest_days", "points_last_5"]


## Recommendation 3 & 6: Optuna with TimeSeriesSplit
def objective(trial):
    param = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 1,
        "objective": "multi:softmax",
        "num_class": 3
    }
    xgb = XGBClassifier(**param)

    tscv = TimeSeriesSplit(n_splits=3)
    precisions = []

    # Sort by date for time series split
    data = matches_rolling.sort_values("date")
    X = data[features]
    y = data["target"]

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)

        # calculate precision macro across multi-class
        precisions.append(precision_score(y_test, preds, average="macro", zero_division=0))

    return sum(precisions) / len(precisions)


# We no longer need to run Optuna dynamically because we discovered the optimal parameters.
# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=200)
# best_params = study.best_params

# Best parameters gathered from a 200 trial run using Optuna:
best_params = {
    "n_estimators": 198,
    "max_depth": 4,
    "learning_rate": 0.01685753063997082,
    "subsample": 0.5929065714918226,
    "colsample_bytree": 0.7837450260638121,
    "random_state": 1,
    "objective": "multi:softmax",
    "num_class": 3
}

## Recommendation 4: Use XGBoost
model = XGBClassifier(**best_params)


def make_predictions(data, predictors):  ## making the predictions
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] >= '2022-01-01']
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors])  ##making prediction
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds, average="macro", zero_division=0)
    return combined, precision  ## returning the values for the prediction


combined, precision = make_predictions(matches_rolling, features)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)


class MissingDict(dict):  ## creating a class that inherits from the dictionary class
    __missing__ = lambda self, key: key  ## in case a team name is missing


map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)

combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date",
                                                                          "opponent"])  ## finding both the home and away team predictions and merging them

# --- ADD THIS TO THE BOTTOM OF YOUR FILE ---

print("\n" + "="*40)
print("🏆 MODEL EVALUATION & STATISTICS 🏆")
print("="*40)

# 1. Print Optuna Best Parameters
print("\n[1] OPTUNA BEST HYPERPARAMETERS:")
for key, value in best_params.items():
    print(f"    {key}: {value}")

# 2. Print Precision & Accuracy Scores
print("\n[2] OVERALL SCORES:")
accuracy = accuracy_score(combined["actual"], combined["prediction"])
print(f"    Accuracy:  {accuracy:.2%}")
print(f"    Precision (Macro): {precision:.2%}")

# 3. Print a Confusion Matrix (Actual vs Predicted)
# 0 = Loss, 1 = Draw, 2 = Win
print("\n[3] CONFUSION MATRIX (Actual vs Prediction):")
crosstab = pd.crosstab(
    index=combined["actual"],
    columns=combined["prediction"],
    rownames=["Actual (0=L, 1=D, 2=W)"],
    colnames=["Predicted (0=L, 1=D, 2=W)"]
)
print(crosstab)

# 4. Evaluate "Strong" Predictions
# Where the model predicted Home to Win (2) AND Away to Lose (0)
print("\n[4] TWO-WAY MERGED MATCHES (Strong Predictions):")
# Filter where model predicted team_x to win and team_y to lose
strong_preds = merged[(merged["prediction_x"] == 2) & (merged["prediction_y"] == 0)]

if len(strong_preds) > 0:
    # Calculate how often these 'strong' predictions were actually correct (team_x actually won)
    strong_accuracy = accuracy_score(strong_preds["actual_x"], strong_preds["prediction_x"])
    print(f"    Found {len(strong_preds)} matches where Model predicted Team A wins and Team B loses.")
    print(f"    Accuracy of these Strong Predictions: {strong_accuracy:.2%}")
else:
    print("    No strong overlapping predictions found.")
print("\n========================================")

# --- 5. Feature Importance Chart ---
import matplotlib.pyplot as plt
import numpy as np

# model is already trained from make_predictions
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

# Display the top 20 most important features
top_n = 20
top_indices = indices[:top_n]
top_features = [features[i] for i in top_indices]
top_importances = importances[top_indices]

plt.figure(figsize=(10, 8))
plt.title("Top 20 Feature Importances (XGBoost)")
plt.barh(range(top_n), top_importances, align="center")
plt.yticks(range(top_n), top_features)
plt.gca().invert_yaxis() # Highest importance at the top
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()
