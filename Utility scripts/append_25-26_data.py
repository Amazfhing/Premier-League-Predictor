"""
Append 25-26data.csv to matches.csv

This script:
1. Reads 25-26data.csv (2025-2026 Premier League data)
2. Reads matches.csv (main dataset)
3. Maps columns from 25-26data.csv to match matches.csv schema
4. Adds default values for missing columns
5. Determines season based on date (2025 for 2025-26 season)
6. Appends the data to matches.csv
"""

import pandas as pd
from datetime import datetime

def determine_season(date_str):
    """
    Determine the season year from a date
    For 2025-26 season, matches from Aug 2025 to May 2026 = 2025
    """
    date = pd.to_datetime(date_str)
    year = date.year
    month = date.month

    # If month is Aug-Dec, season year is current year
    # If month is Jan-May, season year is previous year
    if month >= 8:
        return year
    else:
        return year - 1

def main():
    
    source_file = 'Datasets/25-26data.csv'
    target_file = 'Datasets/matches.csv'

   
    df_source = pd.read_csv(source_file)

    
    df_target = pd.read_csv(target_file, index_col=0)

    
    df_append = pd.DataFrame()

    # Copy existing columns from source
    df_append['date'] = df_source['date']
    df_append['time'] = df_source['time']
    df_append['round'] = df_source['round']
    df_append['day'] = df_source['day']
    df_append['venue'] = df_source['venue']
    df_append['result'] = df_source['result']
    df_append['gf'] = df_source['gf']
    df_append['ga'] = df_source['ga']
    df_append['opponent'] = df_source['opponent']
    df_append['xg'] = df_source['xg']
    df_append['xga'] = df_source['xga']
    df_append['poss'] = df_source['poss']
    df_append['sot'] = df_source['sot']
    df_append['team'] = df_source['team']

    # Add defaults for missing columns
    df_append['comp'] = 'Premier League'
    df_append['attendance'] = ''
    df_append['captain'] = ''
    df_append['formation'] = ''
    df_append['referee'] = ''
    df_append['match report'] = ''
    df_append['notes'] = ''
    df_append['sh'] = None  # Total shots - not available
    df_append['dist'] = None  # Shot distance - not available
    df_append['fk'] = None  # Free kicks - not available
    df_append['pk'] = None  # Penalties scored - not available
    df_append['pkatt'] = None  # Penalties attempted - not available

    # Determine season for each row
    df_append['season'] = df_append['date'].apply(determine_season)


    target_columns = df_target.columns.tolist()
    df_append = df_append[target_columns]

    df_combined = pd.concat([df_target, df_append], ignore_index=False)

    df_combined.to_csv(target_file)

if __name__ == '__main__':
    main()
