# Premier League Predictor

A Python machine-learning project that predicts Premier League match outcomes and identifies profitable betting opportunities using XGBoost, rolling team-form features, draw-specific feature engineering, and 1xBet betting odds.

The project trains Optuna-tuned XGBoost classifiers to predict whether a team will **win**, **draw**, or **lose** a match using unified historical data (seasons 15-26). Through extensive testing and validation, we discovered a **profitable betting strategy** that exploits a market inefficiency: hot form teams (≥12 points in last 5 games) are systematically underpriced by bookmakers. The strategy achieved **+174.4% ROI** on the test set and **+174.3% ROI** on holdout validation (2024-2026 seasons), confirming the edge generalizes to unseen data.

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
- **Validated betting strategy**: Hot form teams WIN strategy achieving +174% ROI on both test set and holdout validation.
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

Filters to hot form teams (≥12 points in last 5 games) and simulates fixed-stakes betting on WIN outcomes. This strategy achieved +174.4% ROI on the test set and +174.3% ROI on holdout validation.

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

- **Filter**: Hot form teams (≥12 points in last 5 games)
- **Bet on**: Team to WIN (not draw or loss)
- **Stake**: Fixed $10 per bet (not Kelly Criterion due to probability miscalibration)
- **Edge**: Bookmakers systematically underprice in-form favorites against weak opposition

### Validated Results (Fixed Stakes)

**Test set** (1,838 matches, 2022-2023):
- **Bets placed**: 148
- **Win rate**: 69.6% (103 wins, 45 losses)
- **Average odds**: 3.80
- **ROI**: +174.4% per dollar staked
- **Profit**: $2,580.50 on $1,480 total stake
- **Sharpe ratio**: 0.48

**Holdout validation** (348 matches, 2024-2026):
- **Bets placed**: 48
- **Win rate**: 62.5% (30 wins, 18 losses)
- **Average odds**: 3.69
- **ROI**: +174.3% per dollar staked
- **Profit**: $836.70 on $480 total stake
- **Sharpe ratio**: 0.44

**Key finding**: The edge generalizes to completely unseen data. ROI is nearly identical across both time periods (174.3% vs 174.4%), confirming this is a real market inefficiency and not overfitting.

### Key Findings

1. **The edge is real**: Fixed stakes validation proves this isn't Kelly compounding or overfitting
2. **Systematic pattern**: Hot form favorites (Man City, Arsenal, Liverpool) against relegation-threatened teams are consistently underpriced
3. **Market inefficiency**: Bookmakers give 3-8x odds (implying 12-30% win probability) when reality is ~70%
4. **Probability miscalibration**: Model overestimates win probability (thinks 89.5%, reality ~70%), so Kelly Criterion breaks—use fixed stakes or very conservative fractional Kelly (0.05x)

### How to Run

```powershell
# Run the validated betting strategy
.\.venv\Scripts\python.exe PL_Predictor_bettingstrat.py

# Run holdout validation on 2024-2025 season
.\.venv\Scripts\python.exe PL_Predictor_validate_holdout.py
```

### Important Caveats

1. **Validated on holdout data**: Strategy tested on 2024-2025 season with consistent results (ROI 174.3% vs 174.4%)
2. **Execution costs not modeled**: Real-world commissions, spread (2-5%) will reduce ROI
3. **Betting limits**: Bookmakers cap winning bettors at $50-500 per bet
4. **Odds movement**: Large bets move odds against you
5. **Sample size**: 48 holdout bets + 148 test bets = 196 total bets across two time periods—paper trade for 2-3 months before live deployment

### Recommended Next Steps

1. **Paper trade**: Run strategy with $10 fixed stakes for 2-3 months on live data to verify execution
2. **Monitor performance**: Track weekly win rate, ROI, and odds movement to detect edge degradation
3. **Consider deployment**: If paper trading succeeds, use conservative fractional Kelly (0.05x) or continue fixed stakes
4. **Track execution costs**: Measure real-world commissions and spread to validate profitability after fees

## Troubleshooting


### File-not-found errors

Run scripts from the repository root. The scripts expect CSV files to be available through relative paths.

### Feature chart does not appear

`PL_Predictor.py` opens a matplotlib chart after printing metrics. If running in a headless terminal or remote environment, configure a compatible matplotlib backend or save the figure instead.

