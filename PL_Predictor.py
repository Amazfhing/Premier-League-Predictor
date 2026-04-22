## PL Predictor using scikit-learn to predict from the matches.csv stat sheet containing data from all matches from 2022-2020
import pandas as pd 
matches = pd.read_csv("matches.csv", index_col = 0)

import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score

##converting all objects to int or float to be processed by the machine learning software
matches["date"] = pd.to_datetime(matches["date"])

## Recommendation 5: Multi-Class Target (Win:2, Draw:1, Loss:0)
result_map = {"L": 0, "D": 1, "W": 2}
matches["target"] = matches["result"].map(result_map)

## Recommendation 2: One-Hot Encoding for categorical features
matches = pd.concat([matches, pd.get_dummies(matches[["venue", "opponent"]], drop_first=True)], axis=1)

matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int") ## converting hours to number in case a team plays better at a certain time
matches["day"] = matches["date"].dt.dayofweek ## converting day of week of game to a number

# Generate a list of predictors dynamically based on one-hot encoding
predictors = ["hour", "day"] + [col for col in matches.columns if col.startswith("venue_") or col.startswith("opponent_")]

grouped_matches = matches.groupby("team") 
group = grouped_matches.get_group("Manchester United").sort_values("date")
 
def rolling_averages(group, cols, new_cols): ## function to take into consideration form of a team
    group = group.sort_values("date") ## sorting games by date 
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols) ##droping missing values and replacing with empty
    return group 

cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"] 
new_cols = [f"{c}_rolling" for c in cols] ## creating new columns with rolling average values 

rolling_averages(group, cols, new_cols) ## calling function and generating average of last 3 games

matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team') ## dropping extra index level

matches_rolling.index = range(matches_rolling.shape[0]) ## adding new index

features = predictors + new_cols

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

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10) # 10 trials for brevity

best_params = study.best_params
best_params["random_state"] = 1
best_params["objective"] = "multi:softmax"
best_params["num_class"] = 3

## Recommendation 4: Use XGBoost
model = XGBClassifier(**best_params)

def make_predictions(data, predictors): ## making the predictions
    train = data[data["date"] < '2022-01-01'] 
    test = data[data["date"] >= '2022-01-01']
    model.fit(train[predictors], train["target"])
    preds = model.predict(test[predictors]) ##making prediction
    combined = pd.DataFrame(dict(actual=test["target"], prediction=preds), index=test.index)
    precision = precision_score(test["target"], preds, average="macro", zero_division=0)
    return combined, precision ## returning the values for the prediction

combined, precision = make_predictions(matches_rolling, features)

combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index = True, right_index = True)


class MissingDict(dict): ## creating a class that inherits from the dictionary class
    __missing__ = lambda self, key: key ## in case a team name is missing

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Tottenham Hotspur": "Tottenham", 
    "West Ham United": "West Ham", 
    "Wolverhampton Wanderers": "Wolves"
}
mapping = MissingDict(**map_values)


combined["new_team"] = combined["team"].map(mapping)

merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"]) ## finding both the home and away team predictions and merging them 

