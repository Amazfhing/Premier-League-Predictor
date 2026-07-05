import pandas as pd

# Read matches data (now includes betting odds: odds_H, odds_D, odds_A)
matches = pd.read_csv("Datasets/matches.csv", index_col=0)

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

matches["date"] = pd.to_datetime(matches["date"])

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



cols = ["gf", "ga", "sot"]

# Prefer xG over actual goals if available (more stable, less noisy)
if "xg" in matches.columns and "xga" in matches.columns:
    cols.extend(["xg", "xga"])

new_cols = [f"{c}_rolling" for c in cols]  # creating new columns with rolling average values
new_cols_10 = [f"{c}_10_rolling" for c in cols]

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')

matches_rolling.index = range(matches_rolling.shape[0])

features = predictors + new_cols + new_cols_10 + ["rest_days", "points_last_5"]


if "odds_H" in matches_rolling.columns:
    features.extend(["odds_H", "odds_D", "odds_A"])

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
    xgb = XGBClassifier(**param) #Feeds the parameters into the actual XGBoost model

    tscv = TimeSeriesSplit(n_splits=3)
    precisions = []

    data = matches_rolling[matches_rolling["date"] < '2022-01-01'].sort_values("date")
    X = data[features]
    y = data["target"]

    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        precisions.append(precision_score(y_test, preds, average="macro", zero_division=0))

    return sum(precisions) / len(precisions)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=200)
best_params = study.best_params
best_params["random_state"] = 1
best_params["objective"] = "multi:softmax"
best_params["num_class"] = 3

# Best parameters gathered from previous Optuna runs

model = XGBClassifier(**best_params)

def make_predictions(data, predictors):
    """
    TimeSeriesSplit for train/validation/test:
    - Test set: 2023-08 onwards (held-out, never used for training)
    - Train/Val: < 2023-08, split using TimeSeriesSplit(n_splits=3)
    - Uses the last fold (most training data) for final model
    """
    # Hold out test set (most recent seasons - never touched during training)
    test_cutoff = '2023-08-01'
    train_val_data = data[data["date"] < test_cutoff].sort_values("date")
    test_data = data[data["date"] >= test_cutoff].sort_values("date")

    # TimeSeriesSplit on train_val_data to create expanding training sets
    tscv = TimeSeriesSplit(n_splits=3)
    X_train_val = train_val_data[predictors]
    y_train_val = train_val_data["target"]

    # Get all splits and use the last one (most data for training)
    splits = list(tscv.split(X_train_val))
    train_idx, val_idx = splits[-1]  # Last split has most training data

    X_train = X_train_val.iloc[train_idx]
    y_train = y_train_val.iloc[train_idx]
    X_val = X_train_val.iloc[val_idx]
    y_val = y_train_val.iloc[val_idx]

    # Train the model on the training fold
    model.fit(X_train, y_train)

    # Validation metrics (on the validation fold)
    val_preds = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_preds)
    val_precision = precision_score(y_val, val_preds, average="macro", zero_division=0)

    # Get date ranges for reporting
    train_dates = train_val_data.iloc[train_idx]["date"]
    val_dates = train_val_data.iloc[val_idx]["date"]

    print(f"\nTimeSeriesSplit Validation (Last Fold of 3):")
    print(f"  Training: {train_dates.min().date()} to {train_dates.max().date()} ({len(X_train)} samples)")
    print(f"  Validation: {val_dates.min().date()} to {val_dates.max().date()} ({len(X_val)} samples)")
    print(f"  Accuracy: {val_accuracy:.2%}")
    print(f"  Precision: {val_precision:.2%}")

    # Test set evaluation (held-out data)
    X_test = test_data[predictors]
    y_test = test_data["target"]
    test_preds = model.predict(X_test)

    combined = pd.DataFrame(
        dict(actual=y_test, prediction=test_preds),
        index=test_data.index
    )
    test_precision = precision_score(y_test, test_preds, average="macro", zero_division=0)

    return combined, test_precision


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

print("\n" + "="*80)
print("OPTIMIZED MODEL EVALUATION (Reduced Feature Set)")
print("="*80)
print(f"\nTotal Features: {len(features)}")
print(f"Accuracy:  {accuracy:.2%} | Precision: {precision:.2%}")
print("\nRemoved redundant features:")
print("  - sh (total shots) - kept sot (shots on target) instead")
print("  - dist (shot distance) - noisy, low predictive value")
print("  - fk (free kicks) - rare events, low signal")
print("  - pk, pkatt (penalties) - rare events, high correlation")
print("\nKept core features:")
print("  - gf, ga (goals for/against) - fundamental metrics")
print("  - sot (shots on target) - quality shooting metric")
print("  - xg, xga (if available) - more stable than actual goals")
print("  - Both 3-game and 10-game rolling (captures short/long-term form)")
print("  - Betting odds (team_odds, draw_odds, opp_odds)")
print("  - Context (venue, opponent, hour, day, rest_days, points_last_5)")

print("\n\nConfusion Matrix (0=Loss, 1=Draw, 2=Win):")

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    combined["actual"],
    combined["prediction"],
    display_labels=["Loss", "Draw", "Win"],
    cmap="Blues",
    ax=ax
)
plt.title("Confusion Matrix - Premier League Predictions (Optimized)")
plt.tight_layout()

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
plt.title("Top 20 Feature Importances (XGBoost - Optimized Model)")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.gca().invert_yaxis()
plt.xlabel("Relative Importance")
plt.tight_layout()
plt.show()

print("\n" + "="*80)
