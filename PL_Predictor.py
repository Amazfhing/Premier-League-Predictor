import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


matches = pd.read_csv("Datasets/matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])

# Convert odds to team perspective
matches["team_odds"] = np.where(matches["venue"] == "Home", matches["odds_H"], matches["odds_A"])
matches["opp_odds"] = np.where(matches["venue"] == "Home", matches["odds_A"], matches["odds_H"])
matches["draw_odds"] = matches["odds_D"]

# Encode results
result_mapping = {"L": 0, "D": 1, "W": 2}
matches["result_code"] = matches["result"].map(result_mapping)
matches["points"] = matches["result_code"].map({0: 0, 1: 1, 2: 3})
matches["is_draw"] = (matches["result"] == "D").astype(int)

# Rolling averages function
def rolling_averages(group, cols, new_cols, window_size=3):
    group = group.sort_values("date")
    group["rest_days"] = group["date"].diff().dt.days.fillna(7)
    rolling_stats = group[cols].shift(1).rolling(window=window_size, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# Feature columns
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
if "xg" in matches.columns and "xga" in matches.columns:
    cols.extend(["xg", "xga"])

new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages
matches_rolling = matches.groupby("team", group_keys=False).apply(
    lambda x: rolling_averages(x, cols, new_cols, window_size=3), include_groups=True
)

# 10-match rolling
cols_10 = cols.copy()
new_cols_10 = [f"{c}_rolling_10" for c in cols_10]
matches_rolling = matches_rolling.groupby("team", group_keys=False).apply(
    lambda x: rolling_averages(x, cols_10, new_cols_10, window_size=10), include_groups=True
)

# Points last 5
def add_points_last_5(group):
    group = group.sort_values("date")
    group["points_last_5"] = group["points"].shift(1).rolling(5, closed='left').sum()
    return group

matches_rolling = matches_rolling.groupby("team", group_keys=False).apply(add_points_last_5, include_groups=True)
matches_rolling = matches_rolling.dropna(subset=["points_last_5"])

# Draw-specific features
def add_draw_features(group):
    group = group.sort_values("date")
    group["gf_rolling"] = group["gf"].shift(1).rolling(5, closed='left').mean()
    group["ga_rolling"] = group["ga"].shift(1).rolling(5, closed='left').mean()
    group["low_scoring"] = ((group["gf_rolling"] < 1.2) & (group["ga_rolling"] < 1.2)).astype(int)
    group["balanced_form"] = ((group["points_last_5"] >= 6) & (group["points_last_5"] <= 9)).astype(int)
    group["recent_draws"] = group["is_draw"].shift(1).rolling(5, closed='left').sum().fillna(0)
    group["goals_variance"] = (group["gf"] - group["ga"]).shift(1).rolling(5, closed='left').std().fillna(0)
    return group

matches_rolling = matches_rolling.groupby("team", group_keys=False).apply(add_draw_features, include_groups=True)

# H2H draw rate
h2h_draws = matches_rolling.groupby(["team", "opponent"])["is_draw"].transform("mean")
matches_rolling["h2h_draw_rate"] = h2h_draws

# Odds-based features
matches_with_odds = matches_rolling.dropna(subset=["team_odds", "draw_odds", "opp_odds"]).copy()
matches_with_odds["odds_balance"] = np.abs(
    matches_with_odds["team_odds"] - matches_with_odds["opp_odds"]
) / matches_with_odds["draw_odds"]
matches_with_odds["draw_value"] = 1 / matches_with_odds["draw_odds"]
matches_with_odds["favorite_strength"] = np.maximum(
    1 / matches_with_odds["team_odds"], 1 / matches_with_odds["opp_odds"]
) - np.minimum(
    1 / matches_with_odds["team_odds"], 1 / matches_with_odds["opp_odds"]
)

# Time features
matches_with_odds["hour"] = matches_with_odds["time"].str.split(':').str[0].astype(int)
matches_with_odds["day"] = matches_with_odds["date"].dt.dayofweek

# One-hot encoding
matches_encoded = pd.get_dummies(matches_with_odds, columns=["venue", "opponent"], dtype=int)

# Build predictors
venue_cols = [col for col in matches_encoded.columns if col.startswith("venue_")]
opponent_cols = [col for col in matches_encoded.columns if col.startswith("opponent_")]
predictors = (
    new_cols + new_cols_10 +
    ["hour", "day", "rest_days", "points_last_5"] +
    ["h2h_draw_rate", "odds_balance", "draw_value", "favorite_strength",
     "low_scoring", "balanced_form", "recent_draws", "goals_variance"] +
    ["team_odds", "draw_odds", "opp_odds"] +
    venue_cols + opponent_cols
)

# Train/test split
train = matches_encoded[matches_encoded["date"] < "2022-01-01"].copy()
test = matches_encoded[matches_encoded["date"] >= "2022-01-01"].copy()

train = train.dropna(subset=predictors)
test = test.dropna(subset=predictors)

# Optuna hyperparameter optimization

def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "random_state": 42,
        "objective": "multi:softprob"
    }

    # TimeSeriesSplit cross-validation on training data
    tscv = TimeSeriesSplit(n_splits=3)
    f1_scores = []

    for train_idx, val_idx in tscv.split(train):
        X_train_fold = train.iloc[train_idx][predictors]
        y_train_fold = train.iloc[train_idx]["result_code"]
        X_val_fold = train.iloc[val_idx][predictors]
        y_val_fold = train.iloc[val_idx]["result_code"]

        # Calculate sample weights for this fold
        fold_weights = compute_sample_weight(class_weight="balanced", y=y_train_fold)

        # Train and evaluate
        model = XGBClassifier(**params)
        model.fit(X_train_fold, y_train_fold, sample_weight=fold_weights, verbose=False)
        preds = model.predict(X_val_fold)
        f1 = f1_score(y_val_fold, preds, average="macro", zero_division=0)
        f1_scores.append(f1)

    return np.mean(f1_scores)

# Run Optuna optimization
study = optuna.create_study(direction="maximize", study_name="draw_features_optimization")
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"Best F1 score: {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best hyperparameters

best_params = study.best_params.copy()
best_params.update({"random_state": 42, "objective": "multi:softprob"})

# Calculate sample weights for full training set
sample_weights = compute_sample_weight(class_weight="balanced", y=train["result_code"])

xgb_model = XGBClassifier(**best_params)
xgb_model.fit(train[predictors], train["result_code"], sample_weight=sample_weights)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
calibrated_model.fit(train[predictors], train["result_code"])

# Predict
y_pred_proba = calibrated_model.predict_proba(test[predictors])
y_pred = calibrated_model.predict(test[predictors])

# Evaluate
accuracy = accuracy_score(test["result_code"], y_pred)
precision = precision_score(test["result_code"], y_pred, average="macro", zero_division=0)
conf_matrix = confusion_matrix(test["result_code"], y_pred)


print("MODEL PERFORMANCE:")
print(f"\nAccuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
# Display confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Loss', 'Draw', 'Win'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Hot Form Teams WIN Strategy')
plt.tight_layout()
plt.show()

avg_probs = y_pred_proba.mean(axis=0)
print(f"\nModel probability estimates:")
print(f"  Loss: {avg_probs[0]:.1%}")
print(f"  Draw: {avg_probs[1]:.1%} (true: {(test['result_code']==1).mean():.1%})")
print(f"  Win:  {avg_probs[2]:.1%}")

# Feature importance
importances = xgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'feature': predictors,
    'importance': importances
}).sort_values('importance', ascending=False).head(20)

plt.figure(figsize=(10, 8))
plt.barh(range(len(feature_importance)), feature_importance['importance'])
plt.yticks(range(len(feature_importance)), feature_importance['feature'])
plt.xlabel('Importance')
plt.title('Top 20 Feature Importances - Draw Features Model')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('draw_features_importance.png', dpi=150, bbox_inches='tight')

