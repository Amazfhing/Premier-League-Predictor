"""
Append 2024-25 season data to matches.csv

This script reads the 24-25data.csv file, filters for the specified date range
(2024-08-16 to 2025-05-25), maps columns to match the matches.csv schema,
and appends the filtered data to the main matches.csv file.
"""

import pandas as pd
from datetime import datetime

def main():
    
    source_file = 'Datasets/24-25data.csv'
    target_file = 'Datasets/matches.csv'

    
    start_date = '2024-08-16'
    end_date = '2025-05-25'

    
    df_source = pd.read_csv(source_file)

    
    df_source['date'] = pd.to_datetime(df_source['date'])

    
    mask = (df_source['date'] >= start_date) & (df_source['date'] <= end_date)
    df_filtered = df_source[mask].copy()

    if len(df_filtered) == 0:
        return

    # Convert date back to string format for consistency with matches.csv
    df_filtered['date'] = df_filtered['date'].dt.strftime('%Y-%m-%d')

    
    df_target = pd.read_csv(target_file, index_col=0)

    # Map columns from source to target 
    # Source has: date, time, round, day, venue, result, gf, ga, opponent, xg, xga, poss, sot, team
    # Target has: date, time, comp, round, day, venue, result, gf, ga, opponent, xg, xga, poss,
    #             attendance, captain, formation, referee, match report, notes, sh, sot, dist, fk, pk, pkatt, season, team

    # Create a new dataframe with the target schema
    df_append = pd.DataFrame()

    # Copy existing columns
    df_append['date'] = df_filtered['date']
    df_append['time'] = df_filtered['time']
    df_append['round'] = df_filtered['round']
    df_append['day'] = df_filtered['day']
    df_append['venue'] = df_filtered['venue']
    df_append['result'] = df_filtered['result']
    df_append['gf'] = df_filtered['gf']
    df_append['ga'] = df_filtered['ga']
    df_append['opponent'] = df_filtered['opponent']
    df_append['xg'] = df_filtered['xg']
    df_append['xga'] = df_filtered['xga']
    df_append['poss'] = df_filtered['poss']
    df_append['sot'] = df_filtered['sot']
    df_append['team'] = df_filtered['team']

    # Add defaults for missing columns
    df_append['comp'] = 'Premier League'
    df_append['attendance'] = ''  # Empty - not available in source
    df_append['captain'] = ''  # Empty - not available in source
    df_append['formation'] = ''  # Empty - not available in source
    df_append['referee'] = ''  # Empty - not available in source
    df_append['match report'] = ''  # Empty - not available in source
    df_append['notes'] = ''  # Empty - not available in source
    df_append['sh'] = None  # NaN - shots total not in source (only sot available)
    df_append['dist'] = None  # NaN - not available in source
    df_append['fk'] = None  # NaN - not available in source
    df_append['pk'] = None  # NaN - not available in source
    df_append['pkatt'] = None  # NaN - not available in source
    df_append['season'] = 2024  # 2024-25 season

    
    target_columns = df_target.columns.tolist()
    df_append = df_append[target_columns]

    
    df_combined = pd.concat([df_target, df_append], ignore_index=False)

    df_combined.to_csv(target_file)

if __name__ == '__main__':
    main()
