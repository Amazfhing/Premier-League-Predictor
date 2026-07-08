"""
Fill xG and xGA values in 15-19data.csv from 2014-2020xGandxGA.csv

This script:
1. Reads 15-19data.csv (has NaN xg/xga values)
2. Reads 2014-2020xGandxGA.csv (has xg/xga data)
3. Matches rows based on date, team, and venue (home/away)
4. Fills in the NaN xg/xga values
5. Handles team name variations
6. Saves the updated file
"""

import pandas as pd
import numpy as np

def normalize_team_name(name):
    """
    Normalize team names to handle variations in team names between different data providers

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
        'Newcastle United': 'Newcastle',
        'Norwich City': 'Norwich',
        'West Bromwich Albion': 'West Brom',
        'Leeds United': 'Leeds',
        'Sheffield United': 'Sheffield Utd',
        'Huddersfield Town': 'Huddersfield',
        'Stoke City': 'Stoke',
        'Swansea City': 'Swansea',
        'Cardiff City': 'Cardiff',
    }

    return name_map.get(name, name)

def map_venue(h_a):
    """
    Map h_a column ('h'/'a') to venue ('Home'/'Away')
    """
    return 'Home' if h_a == 'h' else 'Away'

def main():
    # File paths
    data_file = 'Datasets/15-19data.csv'
    xg_file = 'Datasets/2014-2020xGandxGA.csv'


    df_data = pd.read_csv(data_file)


    df_xg = pd.read_csv(xg_file)

    # Normalize team names in both dataframes
    df_data['team_normalized'] = df_data['team'].apply(normalize_team_name)
    df_xg['team_normalized'] = df_xg['team_name'].apply(normalize_team_name)

    # Map venue in xG data
    df_xg['venue'] = df_xg['h_a'].apply(map_venue)

    # Prepare xG data for merging
    # Select only relevant columns and rename for clarity
    df_xg_merge = df_xg[['date', 'team_normalized', 'venue', 'xG', 'xGA']].copy()
    df_xg_merge.columns = ['date', 'team_normalized', 'venue', 'xg_source', 'xga_source']

    # Merge xG data with main data
    # Merge on date, team, and venue
    df_merged = df_data.merge(
        df_xg_merge,
        left_on=['date', 'team_normalized', 'venue'],
        right_on=['date', 'team_normalized', 'venue'],
        how='left',
        suffixes=('', '_new')
    )

    # Fill NaN values in xg and xga columns
    # Track how many values we're filling
    xg_filled = 0
    xga_filled = 0

    for idx, row in df_merged.iterrows():
        # Fill xg if it's NaN and we have xg_source
        if pd.isna(row['xg']) and pd.notna(row['xg_source']):
            df_merged.at[idx, 'xg'] = row['xg_source']
            xg_filled += 1

        # Fill xga if it's NaN and we have xga_source
        if pd.isna(row['xga']) and pd.notna(row['xga_source']):
            df_merged.at[idx, 'xga'] = row['xga_source']
            xga_filled += 1

    # Drop temporary columns
    columns_to_keep = [
        'date', 'time', 'round', 'day', 'venue', 'result',
        'gf', 'ga', 'opponent', 'xg', 'xga', 'poss', 'sot', 'team'
    ]
    df_final = df_merged[columns_to_keep].copy()

    df_final.to_csv(data_file, index=False)



if __name__ == '__main__':
    main()
