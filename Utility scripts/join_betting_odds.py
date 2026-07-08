"""
Join betting odds from 1xBet15-26.csv into matches.csv

This script:
1. Reads matches.csv (team-perspective format)
2. Reads 1xBet15-26.csv (home/away format with combined matchDate)
3. Parses matchDate to extract date and time
4. Matches odds data to matches based on date, team, opponent, and venue
5. Adds H, D, A betting odds columns to matches.csv
6. Handles team name variations
"""

import pandas as pd
from datetime import datetime

def normalize_team_name(name):
    """
    Normalize team names to handle variations between datasets
    """
    name_map = {
        'Man United': 'Manchester Utd',
        'Manchester United': 'Manchester Utd',
        'Man City': 'Manchester City',
        'Spurs': 'Tottenham',
        'Tottenham Hotspur': 'Tottenham',
        'West Ham United': 'West Ham',
        'Leicester City': 'Leicester',
        'Brighton and Hove Albion': 'Brighton',
        'Brighton & Hove Albion': 'Brighton',
        'Wolverhampton Wanderers': 'Wolves',
        'Newcastle': 'Newcastle United',  
        'Newcastle Utd': 'Newcastle United',
        'Norwich City': 'Norwich',
        'West Bromwich Albion': 'West Brom',
        'Leeds United': 'Leeds',
        'Sheffield United': 'Sheffield Utd',
        'Huddersfield Town': 'Huddersfield',
        'Stoke City': 'Stoke',
        'Swansea City': 'Swansea',
        'Cardiff City': 'Cardiff',
        'Nottingham Forest': 'Nottingham',
        'Nott\'ham Forest': 'Nottingham',
        'Nottm Forest': 'Nottingham',
        'Southampton': 'Southampton',
    }

    return name_map.get(name, name)

def parse_match_date(match_date_str):
    """
    Parse matchDate from 'DD-MM-YY HH:MM' format
    Returns date as 'YYYY-MM-DD' and time as 'HH:MM'
    """
    try:
        dt = datetime.strptime(match_date_str, '%d-%m-%y %H:%M')
        date = dt.strftime('%Y-%m-%d')
        time = dt.strftime('%H:%M')
        return date, time
    except Exception as e:
        return None, None

def main():
    
    matches_file = 'Datasets/matches.csv'
    odds_file = 'Datasets/1xBet15-26.csv'

    
    df_matches = pd.read_csv(matches_file, index_col=0)

    
    df_odds = pd.read_csv(odds_file)

    
    df_odds['parsed_date'], df_odds['parsed_time'] = zip(
        *df_odds['matchDate'].apply(parse_match_date)
    )

    # Remove rows where parsing failed
    df_odds = df_odds[df_odds['parsed_date'].notna()].copy()

    # Normalize team names in both datasets
    df_matches['team_normalized'] = df_matches['team'].apply(normalize_team_name)
    df_matches['opponent_normalized'] = df_matches['opponent'].apply(normalize_team_name)
    df_odds['homeTeam_normalized'] = df_odds['homeTeam'].apply(normalize_team_name)
    df_odds['awayTeam_normalized'] = df_odds['awayTeam'].apply(normalize_team_name)

    # Prepare odds data for merging
    # Create two versions: one for home team perspective, one for away team perspective

    # Home team perspective
    df_odds_home = df_odds[[
        'parsed_date', 'parsed_time', 'homeTeam_normalized', 'awayTeam_normalized',
        'H', 'D', 'A'
    ]].copy()
    df_odds_home.columns = [
        'date', 'time', 'team_normalized', 'opponent_normalized',
        'odds_H', 'odds_D', 'odds_A'
    ]
    df_odds_home['venue'] = 'Home'

    # Away team perspective
    df_odds_away = df_odds[[
        'parsed_date', 'parsed_time', 'awayTeam_normalized', 'homeTeam_normalized',
        'A', 'D', 'H'  # Note: A and H are swapped for away perspective
    ]].copy()
    df_odds_away.columns = [
        'date', 'time', 'team_normalized', 'opponent_normalized',
        'odds_H', 'odds_D', 'odds_A'  # From away team's perspective
    ]
    df_odds_away['venue'] = 'Away'

    # Combine both perspectives
    df_odds_combined = pd.concat([df_odds_home, df_odds_away], ignore_index=True)

    # Merge odds with matches
    # Perform left join to keep all matches
    df_merged = df_matches.merge(
        df_odds_combined,
        left_on=['date', 'team_normalized', 'opponent_normalized', 'venue'],
        right_on=['date', 'team_normalized', 'opponent_normalized', 'venue'],
        how='left',
        suffixes=('', '_odds')
    )


    # Drop temporary normalized columns and time_odds (duplicate)
    columns_to_drop = [
        'team_normalized', 'opponent_normalized',
        'time_odds'  # This is a duplicate of the time column
    ]
    df_merged = df_merged.drop(columns=[col for col in columns_to_drop if col in df_merged.columns])

    
    df_merged.to_csv(matches_file)


if __name__ == '__main__':
    main()
