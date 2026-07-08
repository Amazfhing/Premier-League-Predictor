"""
Holdout Validation: Hot Form Teams WIN Betting Strategy

Trains on all data before 2024-01-01, validates on 2024+ data.

This script validates whether the edge found in the test set (2022-2023 data)
generalizes to completely unseen future seasons.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt


matches = pd.read_csv("Datasets/matches.csv", index_col=0)


# Encode results: L=0, D=1, W=2
result_map = {'L': 0, 'D': 1, 'W': 2}
matches['result_code'] = matches['result'].map(result_map)

# Derive points for rolling features
matches['points'] = matches['result_code'].apply(lambda x: 3 if x == 2 else (1 if x == 1 else 0))


# Convert odds to team perspective based on venue
matches['team_odds'] = matches.apply(
    lambda row: row['odds_H'] if row['venue'] == 'Home' else row['odds_A'], axis=1
)
matches['opp_odds'] = matches.apply(
    lambda row: row['odds_A'] if row['venue'] == 'Home' else row['odds_H'], axis=1
)
matches['draw_odds'] = matches['odds_D']

# === Feature Engineering: Rolling Averages ===
def rolling_averages(group, cols, new_cols, window=3):
    """Generate rolling averages per team"""
    group = group.sort_values('date')
    rolling = group[cols].rolling(window, closed='left').mean()
    group[new_cols] = rolling
    group = group.dropna(subset=new_cols)
    return group

# Performance columns for rolling averages
perf_cols = ['gf', 'ga', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'xg', 'xga']

# Generate 3-match rolling averages
matches_rolling_3 = matches.groupby('team', group_keys=False).apply(
    lambda x: rolling_averages(x, perf_cols, [f"{c}_rolling_3" for c in perf_cols], window=3)
)
print(f"After 3-match rolling: {len(matches_rolling_3)} matches")

# Generate 10-match rolling averages
matches_rolling_10 = matches_rolling_3.groupby('team', group_keys=False).apply(
    lambda x: rolling_averages(x, perf_cols, [f"{c}_rolling_10" for c in perf_cols], window=10)
)
print(f"After 10-match rolling: {len(matches_rolling_10)} matches")


def add_rest_days_and_form(group):
    """Add rest days and points from last 5 matches"""
    group = group.sort_values('date')
    group['date'] = pd.to_datetime(group['date'])
    group['rest_days'] = group['date'].diff().dt.days
    group['points_last_5'] = group['points'].rolling(5, closed='left').sum()
    return group

matches_rolling_10 = matches_rolling_10.groupby('team', group_keys=False).apply(add_rest_days_and_form)
matches_rolling_10 = matches_rolling_10.dropna(subset=['rest_days', 'points_last_5'])
print(f"After rest days and form: {len(matches_rolling_10)} matches")

def calculate_h2h_draw_rate(df, lookback_window=5):
    """Calculate head-to-head draw rate"""
    df = df.sort_values('date').copy()
    df['h2h_draw_rate'] = 0.0

    for idx in df.index:
        team = df.loc[idx, 'team']
        opponent = df.loc[idx, 'opponent']
        current_date = df.loc[idx, 'date']

        h2h_matches = df[
            (df['date'] < current_date) &
            (((df['team'] == team) & (df['opponent'] == opponent)) |
             ((df['team'] == opponent) & (df['opponent'] == team)))
        ].tail(lookback_window)

        if len(h2h_matches) > 0:
            draw_count = (h2h_matches['result'] == 'D').sum()
            df.loc[idx, 'h2h_draw_rate'] = draw_count / len(h2h_matches)

    return df

matches_with_features = calculate_h2h_draw_rate(matches_rolling_10)
print(f"After h2h draw rate: {len(matches_with_features)} matches")

matches_with_features['odds_balance'] = abs(matches_with_features['team_odds'] - matches_with_features['opp_odds'])
matches_with_features['draw_value'] = 1 / matches_with_features['draw_odds']
matches_with_features['favorite_strength'] = matches_with_features[['team_odds', 'opp_odds']].min(axis=1)
matches_with_features['low_scoring'] = (matches_with_features['gf_rolling_3'] + matches_with_features['ga_rolling_3'] < 2.5).astype(int)
matches_with_features['balanced_form'] = abs(matches_with_features['gf_rolling_3'] - matches_with_features['ga_rolling_3']) < 0.5

def add_recent_draws(group):
    group = group.sort_values('date')
    group['recent_draws'] = (group['result'] == 'D').rolling(5, closed='left').sum()
    return group

matches_with_features = matches_with_features.groupby('team', group_keys=False).apply(add_recent_draws)
matches_with_features['recent_draws'] = matches_with_features['recent_draws'].fillna(0)
print(f"After recent draws: {len(matches_with_features)} matches")

# Goals variance
matches_with_features['goals_variance'] = matches_with_features.groupby('team')['gf'].transform(
    lambda x: x.rolling(5, closed='left').std()
).fillna(0)

# Extract hour from time (handles formats like "20:15" or "20:15 (21:15)")
matches_with_features['hour'] = matches_with_features['time'].str.split(':').str[0].astype(int)
# Extract day of week from date column (already a datetime)
matches_with_features['date'] = pd.to_datetime(matches_with_features['date'])
matches_with_features['day'] = matches_with_features['date'].dt.dayofweek
print(f"After time feature extraction: {len(matches_with_features)} matches")


encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
venue_opponent_encoded = encoder.fit_transform(matches_with_features[['venue', 'opponent']])
encoded_df = pd.DataFrame(
    venue_opponent_encoded,
    columns=encoder.get_feature_names_out(['venue', 'opponent']),
    index=matches_with_features.index
)
matches_encoded = pd.concat([matches_with_features, encoded_df], axis=1)

venue_cols = [col for col in matches_encoded.columns if col.startswith('venue_')]
opponent_cols = [col for col in matches_encoded.columns if col.startswith('opponent_')]

rolling_3_cols = [f"{c}_rolling_3" for c in perf_cols]
rolling_10_cols = [f"{c}_rolling_10" for c in perf_cols]

draw_feature_cols = [
    'h2h_draw_rate', 'odds_balance', 'draw_value', 'favorite_strength',
    'low_scoring', 'balanced_form', 'recent_draws', 'goals_variance'
]

feature_cols = (
    venue_cols + opponent_cols +
    rolling_3_cols + rolling_10_cols +
    ['hour', 'day', 'rest_days', 'points_last_5'] +
    ['team_odds', 'draw_odds', 'opp_odds'] +
    draw_feature_cols
)

# Holdout Split: Train on <2024-01-01, Test on >=2024-01-01 
matches_encoded['date'] = pd.to_datetime(matches_encoded['date'])
train_df = matches_encoded[matches_encoded['date'] < '2024-01-01'].copy()
holdout_df = matches_encoded[matches_encoded['date'] >= '2024-01-01'].copy()

print(f"\n=== Holdout Validation Setup ===")
print(f"Training data: {len(train_df)} matches (before 2024-01-01)")
print(f"Holdout data: {len(holdout_df)} matches (2024-01-01 onwards)")
print(f"Holdout date range: {holdout_df['date'].min()} to {holdout_df['date'].max()}")


# Drop rows with NaN in predictor columns (SMOTE doesn't handle NaN)
train_df_clean = train_df.dropna(subset=feature_cols)
holdout_df_clean = holdout_df.dropna(subset=feature_cols)

X_train = train_df_clean[feature_cols]
y_train = train_df_clean['result_code']

# Apply SMOTE to balance classes
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train with best hyperparameters from Optuna(Obtained from PL_Predictor_bettingstrat.py)
best_params = {
    'n_estimators': 124,
    'max_depth': 8,
    'learning_rate': 0.18065186239050293,
    'subsample': 0.8911621408743737,
    'colsample_bytree': 0.9833275793023915,
    'gamma': 0.31889340107468545,
    'min_child_weight': 1,
    'random_state': 42,
    'objective': 'multi:softprob',
    'num_class': 3,
    'tree_method': 'hist',
    'eval_metric': 'mlogloss'
}


base_model = XGBClassifier(**best_params)
base_model.fit(X_train_balanced, y_train_balanced)

# Calibrate probabilities
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_train, y_train)

X_holdout = holdout_df_clean[feature_cols]
y_holdout = holdout_df_clean['result_code']

y_pred_holdout = calibrated_model.predict(X_holdout)
y_pred_proba_holdout = calibrated_model.predict_proba(X_holdout)

# Overall holdout performance
accuracy_holdout = accuracy_score(y_holdout, y_pred_holdout)
precision_holdout = precision_score(y_holdout, y_pred_holdout, average='macro', zero_division=0)

print(f"\n=== Overall Holdout Performance ===")
print(f"Accuracy: {accuracy_holdout:.4f}")
print(f"Macro Precision: {precision_holdout:.4f}")
print(f"\nConfusion Matrix:")
conf_matrix_holdout = confusion_matrix(y_holdout, y_pred_holdout)

# Display confusion matrix visualization
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_holdout, display_labels=['Loss', 'Draw', 'Win'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Holdout Validation (2024-2025)')
plt.tight_layout()
plt.show()


hot_form_mask = holdout_df_clean['points_last_5'] >= 12
hot_form_holdout = holdout_df_clean[hot_form_mask].copy()



if len(hot_form_holdout) == 0:
    print("No hot form teams found in holdout data")
    exit()

# Get predictions for hot form matches
X_hot_holdout = hot_form_holdout[feature_cols]
y_hot_holdout = hot_form_holdout['result_code']
y_pred_hot_proba = calibrated_model.predict_proba(X_hot_holdout)

# Add predictions to hot form dataframe
hot_form_holdout['pred_loss_prob'] = y_pred_hot_proba[:, 0]
hot_form_holdout['pred_draw_prob'] = y_pred_hot_proba[:, 1]
hot_form_holdout['pred_win_prob'] = y_pred_hot_proba[:, 2]

# Bet on WIN outcomes with fixed $10 stakes
stake = 10
hot_form_holdout['bet_on_win'] = True
hot_form_holdout['stake'] = stake
hot_form_holdout['potential_return'] = hot_form_holdout['team_odds'] * stake
hot_form_holdout['actual_result'] = hot_form_holdout['result_code']
hot_form_holdout['bet_won'] = hot_form_holdout['actual_result'] == 2  # WIN

# Calculate returns
hot_form_holdout['return'] = hot_form_holdout.apply(
    lambda row: row['potential_return'] if row['bet_won'] else 0, axis=1
)
hot_form_holdout['profit'] = hot_form_holdout['return'] - hot_form_holdout['stake']

# Results 
total_bets = len(hot_form_holdout)
bets_won = hot_form_holdout['bet_won'].sum()
win_rate = bets_won / total_bets if total_bets > 0 else 0

total_staked = hot_form_holdout['stake'].sum()
total_return = hot_form_holdout['return'].sum()
total_profit = hot_form_holdout['profit'].sum()
roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

avg_odds = hot_form_holdout['team_odds'].mean()

print(f"\n=== Holdout Betting Strategy Results ===")
print(f"Strategy: Bet on WIN for hot form teams (fixed $10 stakes)")
print(f"\nBets placed: {total_bets}")
print(f"Bets won: {bets_won}")
print(f"Win rate: {win_rate:.1%}")
print(f"Average odds: {avg_odds:.2f}")
print(f"\nTotal staked: ${total_staked:,.2f}")
print(f"Total return: ${total_return:,.2f}")
print(f"Total profit: ${total_profit:,.2f}")
print(f"ROI: {roi:+.1f}%")

# Sharpe ratio (assuming 0 risk-free rate)
if len(hot_form_holdout) > 1:
    returns_per_bet = hot_form_holdout['profit'] / hot_form_holdout['stake']
    sharpe = returns_per_bet.mean() / returns_per_bet.std() if returns_per_bet.std() > 0 else 0
    print(f"Sharpe ratio: {sharpe:.2f}")


# === Comparison to Original Test Set ===
print(f"\n=== Comparison to Original Test Set (2022-2023) ===")
print(f"Original test set (2022-2023):")
print(f"  - Bets: 148")
print(f"  - Win rate: 69.6%")
print(f"  - ROI: +174.4%")
print(f"  - Profit: $2,580.50")
print(f"\nHoldout validation (2024-2026):")
print(f"  - Bets: {total_bets}")
print(f"  - Win rate: {win_rate:.1%}")
print(f"  - ROI: {roi:+.1f}%")
print(f"  - Profit: ${total_profit:+.2f}")
