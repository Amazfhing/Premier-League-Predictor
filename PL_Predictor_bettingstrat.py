
import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt


matches = pd.read_csv("Datasets/matches.csv", index_col=0)
matches["date"] = pd.to_datetime(matches["date"])

# Convert odds to team perspective based on venue
# Home match: team_odds = odds_H, opp_odds = odds_A
# Away match: team_odds = odds_A, opp_odds = odds_H
matches["team_odds"] = np.where(
    matches["venue"] == "Home",
    matches["odds_H"],
    matches["odds_A"]
)
matches["opp_odds"] = np.where(
    matches["venue"] == "Home",
    matches["odds_A"],
    matches["odds_H"]
)
matches["draw_odds"] = matches["odds_D"]


matches_with_odds = matches.copy()

# Encode results
result_mapping = {"L": 0, "D": 1, "W": 2}
matches_with_odds["result_code"] = matches_with_odds["result"].map(result_mapping)
matches_with_odds["points"] = matches_with_odds["result_code"].map({0: 0, 1: 1, 2: 3})

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
if "xg" in matches_with_odds.columns and "xga" in matches_with_odds.columns:
    cols.extend(["xg", "xga"])

new_cols = [f"{c}_rolling" for c in cols]

# Apply rolling averages
matches_rolling = matches_with_odds.groupby("team", group_keys=False).apply(
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
matches_rolling["is_draw"] = (matches_rolling["result"] == "D").astype(int)

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

# Odds-based features (only for rows with odds)
matches_with_odds_only = matches_rolling.dropna(subset=["team_odds", "draw_odds", "opp_odds"]).copy()
matches_with_odds_only["odds_balance"] = np.abs(
    matches_with_odds_only["team_odds"] - matches_with_odds_only["opp_odds"]
) / matches_with_odds_only["draw_odds"]
matches_with_odds_only["draw_value"] = 1 / matches_with_odds_only["draw_odds"]
matches_with_odds_only["favorite_strength"] = np.maximum(
    1 / matches_with_odds_only["team_odds"], 1 / matches_with_odds_only["opp_odds"]
) - np.minimum(
    1 / matches_with_odds_only["team_odds"], 1 / matches_with_odds_only["opp_odds"]
)

# Time-based features
# Extract hour from time (handles formats like "20:15" or "20:15 (21:15)")
matches_with_odds_only["hour"] = matches_with_odds_only["time"].str.split(':').str[0].astype(int)
matches_with_odds_only["day"] = matches_with_odds_only["date"].dt.dayofweek

# Preserve opponent name for display (use "opp_name" to avoid collision with "opponent_" prefix)
matches_with_odds_only["opp_name"] = matches_with_odds_only["opponent"]

# One-hot encoding
matches_encoded = pd.get_dummies(matches_with_odds_only, columns=["venue", "opponent"], dtype=int)

# Build predictor columns
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
test = matches_encoded[(matches_encoded["date"] >= "2022-01-01") & (matches_encoded["date"] < "2024-01-01")].copy()

# Drop rows with NaN in predictor columns (SMOTE doesn't handle NaN)
train = train.dropna(subset=predictors)
test = test.dropna(subset=predictors)



def objective(trial):
    """Optuna objective: optimize WIN prediction quality for hot form matches"""
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

    tscv = TimeSeriesSplit(n_splits=3)
    f1_scores = []

    for train_idx, val_idx in tscv.split(train):
        X_train_fold = train.iloc[train_idx][predictors]
        y_train_fold = train.iloc[train_idx]["result_code"]
        X_val_fold = train.iloc[val_idx][predictors]
        y_val_fold = train.iloc[val_idx]["result_code"]
        points_val = train.iloc[val_idx]["points_last_5"]

        # Apply SMOTE to training fold
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_fold, y_train_fold)

        # Train model
        model = XGBClassifier(**params)
        model.fit(X_train_smote, y_train_smote, verbose=False)

        # Predict on validation fold
        preds = model.predict(X_val_fold)

        # Filter to hot form matches only (points_last_5 >= 12) - this is what we bet on
        hot_form_mask = points_val >= 12
        if hot_form_mask.sum() > 0:
            y_hot_form = y_val_fold[hot_form_mask]
            preds_hot_form = preds[hot_form_mask]

            # Calculate F1 score on WIN class (label 2) within hot form matches
            # Convert to binary: WIN (1) vs not WIN (0)
            y_binary = (y_hot_form == 2).astype(int)
            preds_binary = (preds_hot_form == 2).astype(int)
            f1 = f1_score(y_binary, preds_binary, average='binary', zero_division=0)
        else:
            f1 = 0

        f1_scores.append(f1)

    return np.mean(f1_scores)

# Run Optuna optimization
study = optuna.create_study(direction="maximize", study_name="hot_form_wins_optimization")
study.optimize(objective, n_trials=100, show_progress_bar=True)


print(f"Best F1 score (on WIN class within hot form): {study.best_value:.4f}")
print(f"\nBest hyperparameters:")
for key, value in study.best_params.items():
    print(f"  {key}: {value}")

# Train final model with best hyperparameters

best_params = study.best_params.copy()
best_params.update({"random_state": 42, "objective": "multi:softprob"})

# Apply SMOTE to full training set
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_balanced, y_train_balanced = smote.fit_resample(train[predictors], train["result_code"])

# Train final model with optimized hyperparameters
xgb_model = XGBClassifier(**best_params)
xgb_model.fit(X_train_balanced, y_train_balanced)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(xgb_model, method="isotonic", cv="prefit")
calibrated_model.fit(train[predictors], train["result_code"])

# Predict on test set
y_pred_proba = calibrated_model.predict_proba(test[predictors])
y_pred = calibrated_model.predict(test[predictors])

# Evaluation
accuracy = accuracy_score(test["result_code"], y_pred)
precision = precision_score(test["result_code"], y_pred, average="macro", zero_division=0)
conf_matrix = confusion_matrix(test["result_code"], y_pred)


print("HOT FORM TEAMS WIN STRATEGY")
print(f"\nModel Performance:")
print(f"  Accuracy:  {accuracy:.2%}")
print(f"  Precision: {precision:.2%}")


# Display confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['Loss', 'Draw', 'Win'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Hot Form Teams WIN Strategy')
plt.tight_layout()
plt.show()

# Betting Strategy: Hot Form Teams to WIN

test_with_probs = test.copy()
test_with_probs["prob_loss"] = y_pred_proba[:, 0]
test_with_probs["prob_draw"] = y_pred_proba[:, 1]
test_with_probs["prob_win"] = y_pred_proba[:, 2]

# Filter to hot form teams (≥12 points last 5)
hot_form_threshold = 12
hot_form_teams = test_with_probs[test_with_probs["points_last_5"] >= hot_form_threshold].copy()

print(f"\nHot Form Filter: points_last_5 >= {hot_form_threshold}")
print(f"Hot form matches: {len(hot_form_teams)} out of {len(test)} test matches")

# Calculate Expected Value for WIN outcome
hot_form_teams["win_ev"] = (
    hot_form_teams["prob_win"] * (hot_form_teams["team_odds"] - 1) -
    (1 - hot_form_teams["prob_win"])
)

# Filter to positive EV WIN bets
ev_threshold = 0.05  # 5% edge minimum
win_bets = hot_form_teams[hot_form_teams["win_ev"] > ev_threshold].copy()

print(f"\nPositive EV WIN bets: {len(win_bets)} (EV > {ev_threshold:.1%})")

if len(win_bets) == 0:
    print("\n No positive EV WIN bets found for hot form teams.")
else:
    FIXED_STAKE = 10  # Fixed bet size in dollars


    # Simulate betting
    starting_bankroll = 1000
    bankroll = starting_bankroll
    bet_history = []

    for idx, row in win_bets.iterrows():
        bet_size = FIXED_STAKE  

        # Determine outcome
        actual_result = row["result_code"]
        predicted_outcome = "Win"

        if actual_result == 2:  # Win
            profit = bet_size * (row["team_odds"] - 1)
            bankroll += profit
            outcome = "Hit"
        else:  # Loss or Draw
            profit = -bet_size
            bankroll += profit
            outcome = "Miss"

        bet_history.append({
            "match": f"{row['team']} vs {row['opp_name']}",
            "date": row["date"],
            "venue": "Home" if "venue_Home" in row and row["venue_Home"] == 1 else "Away",
            "actual_result": ["Loss", "Draw", "Win"][actual_result],
            "predicted": predicted_outcome,
            "prob_win": row["prob_win"],
            "odds": row["team_odds"],
            "ev": row["win_ev"],
            "stake_pct": FIXED_STAKE / starting_bankroll,  # Fixed stake as % of starting bankroll
            "bet_size": bet_size,
            "profit": profit,
            "bankroll": bankroll,
            "outcome": outcome,
        })

    bet_df = pd.DataFrame(bet_history)

    # Calculate statistics
    total_bets = len(bet_df)
    wins = len(bet_df[bet_df["outcome"] == "Hit"])
    losses = len(bet_df[bet_df["outcome"] == "Miss"])
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_staked = bet_df["bet_size"].sum()
    total_profit = bet_df["profit"].sum()
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    final_bankroll = bankroll
    return_pct = ((final_bankroll - starting_bankroll) / starting_bankroll * 100)

    # Average odds and EV
    avg_odds = bet_df["odds"].mean()
    avg_ev = bet_df["ev"].mean()
    avg_prob = bet_df["prob_win"].mean()

    # Implied probability from odds
    implied_prob = (1 / avg_odds)

    # Sharpe-like ratio
    bet_returns = bet_df["profit"] / bet_df["bet_size"]
    sharpe_ratio = bet_returns.mean() / bet_returns.std() if len(bet_returns) > 1 and bet_returns.std() > 0 else 0

    print(f"\n{'Strategy Performance':^80}")
    print(f"Total bets:        {total_bets}")
    print(f"Wins:              {wins}")
    print(f"Losses:            {losses}")
    print(f"Win rate:          {win_rate:.1%}")
    print(f"\nBankroll:")
    print(f"  Starting:        ${starting_bankroll:.2f}")
    print(f"  Final:           ${final_bankroll:.2f}")
    print(f"  Return:          {return_pct:+.1f}%")
    print(f"\nBetting Stats:")
    print(f"  Total staked:    ${total_staked:.2f}")
    print(f"  Total profit:    ${total_profit:+.2f}")
    print(f"  ROI:             {roi:+.1f}%")
    print(f"  Average odds:    {avg_odds:.2f}")
    print(f"  Average EV:      {avg_ev:.1%}")
    print(f"\nProbabilities:")
    print(f"  Model win prob:  {avg_prob:.1%}")
    print(f"  Implied (odds):  {implied_prob:.1%}")
    print(f"  Edge:            {(avg_prob - implied_prob):.1%}")
    print(f"\nRisk Metrics:")
    print(f"  Sharpe ratio:    {sharpe_ratio:.2f}")
    print(f"  Avg bet size:    ${bet_df['bet_size'].mean():.2f} (${FIXED_STAKE} fixed stake)")

    # Top wins 
    print(f"\n{'Top 5 Wins':^80}")
    top_wins = bet_df.nlargest(5, "profit")
    for i, (_, row) in enumerate(top_wins.iterrows(), 1):
        print(f"{i}. {row['match']:<40} ${row['profit']:>8.2f} "
              f"(odds: {row['odds']:.2f}, prob: {row['prob_win']:.1%})")



    # Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Cumulative profit
    bet_df["cumulative_profit"] = (bet_df["bankroll"] - starting_bankroll)
    axes[0, 0].plot(range(len(bet_df)), bet_df["cumulative_profit"], linewidth=2)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    axes[0, 0].set_title("Cumulative Profit Over Time", fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel("Bet Number")
    axes[0, 0].set_ylabel("Profit ($)")
    axes[0, 0].grid(True, alpha=0.3)

    # Odds distribution
    axes[0, 1].hist(bet_df["odds"], bins=20, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(x=avg_odds, color='r', linestyle='--', linewidth=2, label=f'Mean: {avg_odds:.2f}')
    axes[0, 1].set_title("Distribution of WIN Odds", fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel("Odds")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Profit by bet type (all are wins in this strategy)
    axes[1, 0].bar(["WIN Bets"], [total_profit], color='green' if total_profit > 0 else 'red', alpha=0.7)
    axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[1, 0].set_title("Total Profit by Prediction Type", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel("Profit ($)")
    axes[1, 0].grid(True, alpha=0.3, axis='y')


    plt.tight_layout()
    plt.savefig("hot_form_wins_analysis.png", dpi=150, bbox_inches='tight')
    plt.show()
