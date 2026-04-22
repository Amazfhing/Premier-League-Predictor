# Premier-League-Predictor

🛠️ Tech Stack

Language: Python
Data Manipulation: pandas (Core dataframe handling, merging, rolling time-window calculations)
Machine Learning Core: scikit-learn (Metrics formatting, test/train splitting), xgboost (Advanced gradient-boosted decision trees algorithm)
Hyperparameter Tuning: optuna (Bayesian optimization framework)
Data Visualization: matplotlib, numpy (Feature importance bar charts)
 
📂 Project Files breakdown

1. PL_Predictor.py (The Main Engine)
This is the core pipeline of the project. It handles:
Data Ingestion & Cleaning: Reads matches.csv, parses dates, and handles multi-class targets (Win=2, Draw=1, Loss=0).
Feature Engineering: Creates dynamic one-hot encoded variables (for venues and opponents).
Time-Series Logic: Groups data by team and injects "Form" metrics (3-game/10-game rolling averages) to simulate historical momentum.
Odds Integration: Intelligently merges in the normalized Vegas odds.
Model Training & Execution: Uses a pre-optimized XGBClassifier to predict future match outcomes on sequential test data (post-2022).
Evaluation Dashboard: Outputs a concise console report showing overall accuracy, precision, a Confusion Matrix, and isolating high-confidence "Strong Predictions" where both Home and Away models agree, followed by a Matplotlib bar chart showing exactly which features the model values most.
2. MergeOdds.py (The Data Pre-processor)
A standalone utility script that takes raw, messy seasonal data downloads from Football-Data.co.uk (e.g., E0_2021.csv, E0_2122.csv) and does the heavy lifting to combine them into cleaned_odds.csv. It actively scans and checks for team-naming mismatches between the odds data and the FBRef match data, preventing silent merge failures in the main engine.
 
📊 How the Metrics Were Utilized

The model relies entirely on Historical Context and Market Intelligence rather than arbitrary guesses.

1. Form & Momentum Metrics (The "Past")
Rolling Averages (gf_10_rolling, xga_10_rolling): Instead of looking at a team's total points for the year, the model calculates rolling 3-game and 10-game averages for goals, shots, expected goals (xG), and points. This teaches the model about "recent form" (e.g., did they just score 10 goals in 3 games, or are they slumping?).
Rest Days (rest_days): Uses dataframe .diff() to count the days elapsed since a team's previous game to detect fatigue (like a team playing after a short Wednesday European turnaround).
2. Categorical Logic
Opponent & Venue (venue_Home, opponent_Manchester City): One-hot encoded booleans that teach the model the inherent difficulties of the league. It recognizes that playing away against Man City requires a different predictive baseline than playing at home against Norwich.
3. Vegas Betting Markets (The "Market Intelligence")
Odds (team_odds, opp_odds, draw_odds): We integrated the closing lines from Bet365. Vegas bookmakers process infinite unquantifiable real-world data (injuries, manager firings, morale) to set lines. By passing these directly into the XGBoost algorithm, it essentially acts as an expert "baseline cheat sheet", elevating the model's accuracy up to ~57% and allowing it to identify high-confidence (Strong Prediction) disparities efficiently.
