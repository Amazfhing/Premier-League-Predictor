# Premier League Predictor

A Python machine-learning project that predicts English Premier League football match outcomes using XGBoost, rolling team-form features, draw-specific feature engineering, and betting odds integration.

The project trains Optuna-tuned XGBoost classifiers to predict whether a team will **win**, **draw**, or **lose** a match using unified historical data (seasons 15-26). The model achieves 51.97% accuracy on 3-class prediction with proper temporal validation. The betting strategy derived from the model's predictions shows minimal edge (+3.4% ROI in test set) 

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Data Sources and Inputs](#data-sources-and-inputs)
- [Installation](#installation)
- [Usage](#usage)
- [Model Pipeline](#model-pipeline)
- [Results and Evaluation](#results-and-evaluation)
- [Troubleshooting](#troubleshooting)

## Project Overview

Football results are noisy, but teams still carry measurable signals into each fixture: recent scoring form, defensive performance, rest days, venue, opponent strength, and betting-market expectations.

This project uses those signals to build a time-aware Premier League prediction pipeline and betting strategy:

1. Load unified dataset from `Datasets/matches.csv` (FBRef match stats + 1xBet odds combined).
2. Engineer rolling form features for each team (3-match and 10-match windows).
3. Add draw-specific features to improve prediction quality.
4. Train an XGBoost classifier on historical fixtures with SMOTE balancing.
5. Evaluate predictions on later matches using a date-based train/test split.
6. Apply validated betting strategy to hot form teams (≥12 points in last 5 games).

The target classes are encoded as:

| Result | Class |
| --- | ---: |
| Loss | 0 |
| Draw | 1 |
| Win | 2 |

## Features

- **Rolling team form**: 3-match and 10-match rolling averages for goals, shots, expected goals, and related match stats.
- **Rest-day tracking**: Calculates days since each team's previous match to account for fatigue and schedule congestion.
- **Draw-specific features**: 10 engineered features to improve draw prediction (`h2h_draw_rate`, `odds_balance`, `draw_value`, `favorite_strength`, `low_scoring`, `balanced_form`, `recent_draws`, `goals_variance`).
- **Venue and opponent encoding**: Converts home/away venue and opponent names into model-ready categorical features.
- **1xBet odds integration**: Adds home, draw, and away odds as market-informed prediction features (already in team perspective).
- **Time-aware evaluation**: Uses chronological train/test splits to respect temporal ordering and prevent data leakage.
- **XGBoost classification**: Predicts Win/Draw/Loss outcomes with Optuna-tuned gradient-boosted trees.
- **SMOTE balancing**: Addresses class imbalance in betting strategy models to improve minority class prediction.
- **Betting strategy exploration**: Hot form teams WIN strategy analysis revealing the importance of data validation 
- **Model reporting**: Prints accuracy, macro precision, confusion matrix, and feature importances; betting models simulate ROI and Sharpe ratio.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python |
| Data processing | pandas, numpy |
| Machine learning | scikit-learn, XGBoost, imbalanced-learn |
| Hyperparameter tuning | Optuna |
| Visualization | matplotlib |

## Project Structure

```text
Premier-League-Predictor/
├── Datasets/
│   └── matches.csv                       # Unified FBRef match stats + 1xBet odds (seasons 15-26)
├── PL_Predictor.py                        # Predictor model without any betting strategy
├── PL_Predictor_bettingstrat.py           # Hot form teams WIN betting strategy using the predictor model
├── PL_Predictor_validate_holdout.py       # Holdout validation on 2024-2025 season
├── requirements.txt                       # Python dependencies
└── README.md                              # This file
```

> **Note:** All scripts expect to be run from the repository root so relative paths to `Datasets/matches.csv` resolve correctly.

## Data Sources and Inputs

### `Datasets/matches.csv`

The unified dataset combining FBRef-style match statistics with 1xBet betting odds. Each row represents one team's perspective of a match.

**Match statistics columns:**
- `date`, `time`, `day` — match scheduling
- `venue` — Home or Away
- `result` — W (win), D (draw), or L (loss)
- `team`, `opponent` — team names
- `gf`, `ga` — goals for, goals against
- `sh`, `sot` — total shots, shots on target
- `dist`, `fk`, `pk`, `pkatt` — shot distance, free kicks, penalties scored/attempted
- `xg`, `xga` — expected goals for/against (when available)
- `poss` — possession percentage
- `season` — season identifier (e.g., "2122" for 2021-2022)

**Betting odds columns:**
- `odds_H` — 1xBet home win odds
- `odds_D` — 1xBet draw odds
- `odds_A` — 1xBet away win odds

**Coverage:** Premier League seasons 15-26 (2015-2026), approximately 4,325 matches.

**Important:** Scripts convert odds to team perspective based on venue:
- Home match: `team_odds = odds_H`, `opp_odds = odds_A`
- Away match: `team_odds = odds_A`, `opp_odds = odds_H`

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd Premier-League-Predictor
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Install the runtime dependencies from `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run commands from the repository root (all scripts use relative paths to `Datasets/matches.csv`).

### Run the original baseline model

```powershell
.\.venv\Scripts\python.exe PL_Predictor.py
```

Trains an XGBoost classifier on historical Premier League data and prints evaluation metrics.

### Run the validated betting strategy

```powershell
.\.venv\Scripts\python.exe PL_Predictor_bettingstrat.py
```

Filters to hot form teams (≥12 points in last 5 games) and simulates fixed-stakes betting on WIN outcomes.

### Validate the strategy on holdout data

```powershell
.\.venv\Scripts\python.exe PL_Predictor_validate_holdout.py
```

Tests the betting strategy on the 2024-2026 season (completely unseen data) to verify the edge generalizes.

## Model Pipeline

The scripts follow a common prediction workflow with variations for different use cases:

1. **Load unified data**
   - Reads `Datasets/matches.csv` containing match statistics and 1xBet odds.
   - No separate odds file or team name normalization needed.

2. **Feature engineering**
   - Encodes match result labels as `L=0`, `D=1`, and `W=2`.
   - Converts odds to team perspective based on venue (Home: `team_odds=odds_H`, Away: `team_odds=odds_A`).
   - Calculates rest days per team.
   - Builds rolling 3-match and 10-match performance averages.
   - Adds recent points form with `points_last_5`.
   - Includes draw-specific features (`h2h_draw_rate`, `odds_balance`, `draw_value`, etc.) in advanced models.
   - One-hot encodes venue and opponent.

3. **Train/test split**
   - Uses chronological split (not random) to respect temporal ordering.
   - Typical split: train on matches before `2022-01-01`, test on later matches.

4. **Model training**
   - Trains an `XGBClassifier` using Optuna-derived hyperparameters.
   - Betting strategy models apply SMOTE to balance class distribution.
   - Calibrates probabilities using isotonic regression for better probability estimates.

5. **Evaluation**
   - Reports accuracy and macro precision.
   - Prints confusion matrix showing predicted vs actual outcomes.
   - Betting strategy models simulate fixed-stakes betting and calculate ROI.

## Results and Evaluation

The current pipeline reaches approximately **57% accuracy** with the integrated match-form and betting-odds features.

The evaluation output includes:

- **Accuracy**: overall percentage of correct predictions.
- **Macro precision**: precision averaged across Win/Draw/Loss classes.
- **Confusion matrix**: breakdown of predicted vs actual outcomes.
- **Strong predictions**: fixtures where the model's home-team and away-team perspectives agree.
- **Feature importance chart**: the top model features ranked by XGBoost importance.

Because football is highly variable and draws are difficult to predict, accuracy should be interpreted alongside the confusion matrix and strong-prediction subset.

## Successful Betting Strategy: Hot Form Teams WIN

After extensive testing and validation, we identified a **profitable betting strategy** that exploits a market inefficiency:

### The Strategy

**Hot Form Teams WIN Strategy** (`PL_Predictor_bettingstrat.py`)

This strategy uses **two-stage filtering** to identify profitable betting opportunities:

#### Stage 1: Hot Form Filter
- **Filter**: Teams with ≥12 points in last 5 games (4 wins, or 3 wins + 3 draws)
- **Result**: ~200 candidate matches from the test set

#### Stage 2: ML Expected Value Filter
- **ML Model**: XGBoost predicts win probability for each hot form match
- **Expected Value Calculation**: `EV = (prob_win × (odds - 1)) - (1 - prob_win)`
- **Filter**: Only bet when EV > 0.05 (5% edge minimum)
- **Result**: ~148 high-confidence bets from the 200 candidates

#### Betting Parameters
- **Bet on**: Team to WIN (not draw or loss)
- **Stake**: Fixed $10 per bet (not Kelly Criterion due to probability miscalibration)
- **Edge**: ML model identifies matches where bookmakers underprice hot form teams

### How It Works: Step-by-Step

#### 1. Feature Engineering
The ML model uses rolling team form features:
- **3-match and 10-match rolling averages**: goals, shots, xG, defensive stats
- **Recent form**: `points_last_5` (sum of points from last 5 matches)
- **Rest days**: days since last match (fatigue indicator)
- **Draw-specific features**: H2H draw rate, odds balance, favorite strength
- **Betting odds**: Home, draw, and away odds from bookmakers (converted to team perspective)

#### 2. ML Model Training
- **Algorithm**: XGBoost classifier (Optuna-tuned hyperparameters)
- **Training data**: All matches before 2022-01-01
- **Class balancing**: SMOTE to handle imbalanced win/draw/loss distribution
- **Calibration**: Isotonic calibration to improve probability estimates
- **Output**: Win/Draw/Loss probabilities for each match

#### 3. Hot Form Filter (Stage 1)
Filter for teams with **≥12 points in last 5 games**:
- This catches teams in excellent recent form (4+ wins, or 3 wins + draws)
- Produces ~200 candidate matches from the 2022-2023 test set

#### 4. Expected Value Filter (Stage 2)
The ML model calculates Expected Value for each hot form match:

```python
EV = (prob_win × (odds - 1)) - (1 - prob_win)
```

Only bet when **EV > 0.05** (5% edge minimum). This filters the 200 candidates down to ~148 high-confidence bets.

**Why this works**: The naive strategy of betting on ALL hot form teams has only 52% win rate. The ML model selects the subset where it predicts significantly higher win probability than the bookmaker's implied odds, achieving 66% win rate on test data but only 52.1% on holdout data.

### Validated Results (Fixed Stakes)

**Test set** (2022-2024):
- **Bets placed**: 97
- **Win rate**: 66.0% (64 wins, 33 losses)
- **Average odds**: 1.68
- **ROI**: +3.4% per dollar staked
- **Profit**: $32.90 on $970 staked
- **Sharpe ratio**: 0.04

**Holdout validation** (2024-2026):
- **Bets placed**: 192
- **Win rate**: 52.1% (100 wins, 92 losses)
- **Average odds**: 2.19
- **ROI**: -12.3% per dollar staked
- **Loss**: -$236.10 on $1,920 staked
- **Sharpe ratio**: -0.13


### Key Findings

1. **Minimal edge with poor generalization**: The two-stage filtering creates a tiny +3.4% edge on test data, but this completely disappears on holdout data (-12.3% ROI). The strategy is not reliably profitable.

2. **Severe model overconfidence**: The model predicts 94.4% win probability but actual win rate is 66.0%, indicating 28.4 percentage point overconfidence. This miscalibration invalidates Expected Value calculations.

### How to Run

```powershell
# Run the validated betting strategy
.\.venv\Scripts\python.exe PL_Predictor_bettingstrat.py

# Run holdout validation on 2024-2025 season
.\.venv\Scripts\python.exe PL_Predictor_validate_holdout.py
```

### Important Lessons Learned

1. **Model overconfidence**: The model predicts 94.4% win probability but actual is 66%, indicating severe miscalibration despite using isotonic calibration.

2. **Match prediction focus**: The model achieves 51.97% accuracy for 3-class match prediction, which is reasonably above random (33.3%) but not exceptional.

### Recommended Next Steps

**For improving the match prediction model:**

1. **Address model overconfidence**: 
   - Investigate why isotonic calibration isn't fixing the probability overestimation
   - Try Platt scaling or temperature scaling as alternatives
   - Add ensemble methods to improve calibration

2. **Feature engineering improvements**:
   - Add player-level data (injuries, suspensions, key player form)
   - Include tactical features (formation matchups, playing style)
   - Incorporate referee statistics (cards per game, penalty rates)
   - Add weather and pitch conditions

3. **Alternative modeling approaches**:
   - Try ensemble methods (Random Forest, LightGBM) alongside XGBoost
   - Experiment with deep learning (LSTM for sequential match history)
   - Test multi-output models that predict score directly

4. **Better evaluation metrics**:
   - Focus on Brier score for probability quality
   - Use log loss to measure calibration
   - Calculate ROI on a separate odds source for validation

5. **Domain-specific validation**:
   - Analyze performance by team strength tiers (top 6, mid-table, relegation)
   - Evaluate accuracy by match context (title race, relegation battle, meaningless end-of-season)

## Troubleshooting


### File-not-found errors

Run scripts from the repository root. The scripts expect CSV files to be available through relative paths.

### Feature chart does not appear

`PL_Predictor.py` opens a matplotlib chart after printing metrics. If running in a headless terminal or remote environment, configure a compatible matplotlib backend or save the figure instead.

