"""
Fix Newcastle team name mismatch and rejoin data

This script:
1. Checks current Newcastle naming in matches.csv
2. Standardizes all Newcastle variations to "Newcastle United"
3. Fixes 15-19data.csv and 25-26data.csv
4. Re-runs the betting odds join with corrected team name mapping
"""

import pandas as pd

def main():
    # File paths
    matches_file = 'Datasets/matches.csv'
    data_15_19_file = 'Datasets/15-19data.csv'
    data_25_26_file = 'Datasets/25-26data.csv'

    
    df_matches = pd.read_csv(matches_file, index_col=0)


    # Standard name we'll use
    standard_name = "Newcastle United"


    # Replace in team column
    df_matches['team'] = df_matches['team'].replace({
        'Newcastle': standard_name,
        'Newcastle Utd': standard_name,
        'Newcastle United': standard_name
    })

    # Replace in opponent column
    df_matches['opponent'] = df_matches['opponent'].replace({
        'Newcastle': standard_name,
        'Newcastle Utd': standard_name,
        'Newcastle United': standard_name
    })

    # Save updated matches.csv
    df_matches.to_csv(matches_file)

    # Fix 15-19data.csv 
    try:
        df_15_19 = pd.read_csv(data_15_19_file)

        df_15_19['team'] = df_15_19['team'].replace({
            'Newcastle': standard_name,
            'Newcastle Utd': standard_name
        })
        df_15_19['opponent'] = df_15_19['opponent'].replace({
            'Newcastle': standard_name,
            'Newcastle Utd': standard_name
        })

        df_15_19.to_csv(data_15_19_file, index=False)
    except FileNotFoundError:
        pass

    #Fix 25-26data.csv 
    try:
        df_25_26 = pd.read_csv(data_25_26_file)

        df_25_26['team'] = df_25_26['team'].replace({
            'Newcastle': standard_name,
            'Newcastle Utd': standard_name
        })
        df_25_26['opponent'] = df_25_26['opponent'].replace({
            'Newcastle': standard_name,
            'Newcastle Utd': standard_name
        })

        df_25_26.to_csv(data_25_26_file, index=False)
    except FileNotFoundError:
        pass

    # Remove old betting odds columns 
    odds_columns = ['odds_H', 'odds_D', 'odds_A']
    columns_to_drop = [col for col in odds_columns if col in df_matches.columns]

    if columns_to_drop:
        df_matches = df_matches.drop(columns=columns_to_drop)
        df_matches.to_csv(matches_file)

if __name__ == '__main__':
    main()
