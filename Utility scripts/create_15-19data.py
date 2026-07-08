"""
Create 15-19data.csv by merging Goals and SOT/Possession data

This script:
1. Reads Goalsmetric15-20 and 24-25.csv and SOT&Possesion15-20 and 24-25.csv
2. Merges them on common keys (id, matchDate, homeTeam, awayTeam)
3. Filters for dates: August 8, 2015 to May 19, 2019
4. Splits matchDate into 'date' and 'time' columns
5. Transforms from home/away format to team-perspective format
6. Keeps only columns needed for the optimized model
7. Saves as 15-19data.csv
"""

import pandas as pd
from datetime import datetime

def parse_match_date(match_date_str):
    """
    Parse matchDate from format 'DD-MM-YY HH:MM' to date and time
    Example: '08-08-15 13:45' -> ('2015-08-08', '13:45')
    """
    try:
        dt = datetime.strptime(match_date_str, '%d-%m-%y %H:%M')
        date = dt.strftime('%Y-%m-%d')
        time = dt.strftime('%H:%M')
        day = dt.strftime('%a')  # Day of week abbreviation (Mon, Tue, etc.)
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
    """
    Extract round/matchweek info from Season string if possible
    For now, we'll leave this empty as the source data doesn't have matchweek info
    """
    return ''

def main():
    
    goals_file = 'Datasets/Goalsmetric15-20 and 24-25.csv'
    sot_poss_file = 'Datasets/SOT&Possesion15-20 and 24-25.csv'
    output_file = 'Datasets/15-19data.csv'

    # Date range for filtering
    start_date = '2015-08-08'
    end_date = '2019-05-19'

    
    df_goals = pd.read_csv(goals_file)
    df_sot_poss = pd.read_csv(sot_poss_file)

    df_merged = pd.merge(
        df_goals,
        df_sot_poss,
        on=['id', 'matchDate', 'Country', 'League', 'Season', 'homeTeam', 'awayTeam'],
        how='inner'
    )

    # Parse matchDate and filter by date range
    df_merged['parsed_date'], df_merged['parsed_time'], df_merged['parsed_day'] = zip(
        *df_merged['matchDate'].apply(parse_match_date)
    )

    # Remove rows where date parsing failed
    df_merged = df_merged[df_merged['parsed_date'].notna()].copy()

    # Filter by date range
    mask = (df_merged['parsed_date'] >= start_date) & (df_merged['parsed_date'] <= end_date)
    df_filtered = df_merged[mask].copy()

    if len(df_filtered) == 0:
        return

    # Transform from home/away format to team-perspective format
    rows = []

    for _, match in df_filtered.iterrows():
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
            'xg': None,  # Not available in source data
            'xga': None,  # Not available in source data
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
            'xg': None,  # Not available in source data
            'xga': None,  # Not available in source data
            'poss': match['ABPFT'],  # Away Ball Possession Full Time
            'sot': match['ASONFT'],  # Away Shots On target Full Time
            'team': match['awayTeam']
        }
        rows.append(away_row)

    df_final = pd.DataFrame(rows)

    # Ensure column order matches the expected format
    column_order = [
        'date', 'time', 'round', 'day', 'venue', 'result',
        'gf', 'ga', 'opponent', 'xg', 'xga', 'poss', 'sot', 'team'
    ]
    df_final = df_final[column_order]

    # Save to CSV
    df_final.to_csv(output_file, index=False)

if __name__ == '__main__':
    main()
