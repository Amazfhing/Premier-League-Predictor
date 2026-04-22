import pandas as pd

file_1="E0_2021.csv"
file_2="E0_2122.csv"

# Load new Football-Data CSVs
odds_2021 = pd.read_csv(file_1)
odds_2122 = pd.read_csv(file_2)

# Combine them into one big table
all_odds = pd.concat([odds_2021, odds_2122])

# Convert Date column to match pandas datetime format
all_odds["Date"] = pd.to_datetime(all_odds["Date"], dayfirst=True)

# Keep ONLY the columns we actually care about
all_odds = all_odds[["Date", "HomeTeam", "AwayTeam", "B365H", "B365D", "B365A"]]

# Save this cleaned version
all_odds.to_csv("cleaned_odds.csv", index=False)
print("Success: 'cleaned_odds.csv' has been generated.")

# --- TEAM NAME COMPARISON LOGIC ---
matches_df = pd.read_csv("matches.csv")

# Get unique teams from both files
match_teams = set(matches_df["team"].dropna().unique())
odds_teams = set(all_odds["HomeTeam"].dropna().unique())

# Find the mismatched names using Set differences
only_in_matches = sorted(list(match_teams - odds_teams))
only_in_odds = sorted(list(odds_teams - match_teams))

if only_in_matches or only_in_odds:
    print("\nMismatched Teams Detected (Requires Mapping):")
    if only_in_matches:
        print(f"Matches ONLY: {', '.join(only_in_matches)}")
    if only_in_odds:
        print(f"Odds ONLY: {', '.join(only_in_odds)}")
