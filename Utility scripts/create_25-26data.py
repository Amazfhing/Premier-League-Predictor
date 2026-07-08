"""
Create 25-26data.csv by merging Goals, SOT/Possession, and xG data

This script:
1. Reads Goalsmetric25-26.csv, SOT&Possesion25-26.csv, and xG25-26.csv
2. Merges them on common keys (id, matchDate, homeTeam, awayTeam)
3. Splits matchDate into 'date' and 'time' columns
4. Transforms from home/away format to team-perspective format
5. Keeps only columns needed for the optimized model
6. Saves as 25-26data.csv
"""

import pandas as pd
from datetime import datetime

def parse_match_date(match_date_str):
    """
    Parse matchDate from format 'DD-MM-YY HH:MM' to date and time
    Example: '15-08-25 21:00' -> ('2025-08-15', '21:00')
    """
    try:
        dt = datetime.strptime(match_date_str, '%d-%m-%y %H:%M')
        date = dt.strftime('%Y-%m-%d')
        time = dt.strftime('%H:%M')
        day = dt.strftime('%a')  # Day of week abbreviation
        return date, time, day
    except Exception as e:
        return None, None, None

def determine_result(ftr, is_home_team):
    """
    Determine result from team perspective
    FTR: H (Home win), A (Away win), D (Draw)
    """
    if ftr == 'H':
        return 'W' if is_home_team else 'L'
    elif ftr == 'A':
        return 'L' if is_home_team else 'W'
    else:  # Draw
        return 'D'

def extract_round(season_str):
    """Extract round/matchweek - leave empty as source doesn't have it"""
    return ''

def main():
    
    goals_file = 'Datasets/Goalsmetric25-26.csv'
    sot_poss_file = 'Datasets/SOT&Possesion25-26.csv'
    xg_file = 'Datasets/xG25-26.csv'
    output_file = 'Datasets/25-26data.csv'

    
    df_goals = pd.read_csv(goals_file)
    df_sot_poss = pd.read_csv(sot_poss_file)
    df_xg = pd.read_csv(xg_file)

    
    df_merged = pd.merge(
        df_goals,
        df_sot_poss,
        on=['id', 'matchDate', 'Country', 'League', 'Season', 'homeTeam', 'awayTeam'],
        how='inner'
    )

    df_merged = pd.merge(
        df_merged,
        df_xg,
        on=['id', 'matchDate', 'Country', 'League', 'Season', 'homeTeam', 'awayTeam'],
        how='inner'
    )

    # Parse matchDate
    df_merged['parsed_date'], df_merged['parsed_time'], df_merged['parsed_day'] = zip(
        *df_merged['matchDate'].apply(parse_match_date)
    )

    
    df_merged = df_merged[df_merged['parsed_date'].notna()].copy()

    # Transform from home/away format to team-perspective format
    rows = []

    for _, match in df_merged.iterrows():
        date = match['parsed_date']
        time = match['parsed_time']
        day = match['parsed_day']
        round_info = extract_round(match['Season'])

        # Home team row
        home_row = {
            'date': date,
            'time': time,
            'round': round_info,
            'day': day,
            'venue': 'Home',
            'result': determine_result(match['FTR'], is_home_team=True),
            'gf': match['FTHG'],  # Full Time Home Goals
            'ga': match['FTAG'],  # Full Time Away Goals
            'opponent': match['awayTeam'],
            'xg': match['homeXGFT'],  # Home xG Full Time
            'xga': match['awayXGFT'],  # Away xG Full Time (opponent's xG)
            'poss': match['HBPFT'],  # Home Ball Possession Full Time
            'sot': match['HSONFT'],  # Home Shots On target Full Time
            'team': match['homeTeam']
        }
        rows.append(home_row)

        # Away team row
        away_row = {
            'date': date,
            'time': time,
            'round': round_info,
            'day': day,
            'venue': 'Away',
            'result': determine_result(match['FTR'], is_home_team=False),
            'gf': match['FTAG'],  # Full Time Away Goals
            'ga': match['FTHG'],  # Full Time Home Goals
            'opponent': match['homeTeam'],
            'xg': match['awayXGFT'],  # Away xG Full Time
            'xga': match['homeXGFT'],  # Home xG Full Time (opponent's xG)
            'poss': match['ABPFT'],  # Away Ball Possession Full Time
            'sot': match['ASONFT'],  # Away Shots On target Full Time
            'team': match['awayTeam']
        }
        rows.append(away_row)

    df_final = pd.DataFrame(rows)

    # Ensure column order
    column_order = [
        'date', 'time', 'round', 'day', 'venue', 'result',
        'gf', 'ga', 'opponent', 'xg', 'xga', 'poss', 'sot', 'team'
    ]
    df_final = df_final[column_order]

    df_final.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
