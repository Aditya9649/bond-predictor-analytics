import pandas as pd
import requests
import time

# =====================================================================
# YOUR FRED API KEY PLUGGED IN
# =====================================================================
FRED_API_KEY = 'ffcea1d595a807da0f7b5ca32aff856e'

# The two macro series we need
SERIES = {
    'macro_10y_treasury_rate': 'GS10',
    'macro_fed_funds_rate': 'FEDFUNDS'
}

def fetch_fred_series(series_id, api_key):
    '''Downloads one data series API'''
    url = 'https://api.stlouisfed.org/fred/series/observations'
    
    params = {
        'series_id': series_id,
        'api_key': api_key,
        'file_type': 'json',
        'observation_start': '2010-01-01',
        'observation_end': '2026-12-31',
        'frequency': 'm'
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        obs = data['observations']
        df = pd.DataFrame(obs)
        # Keep only date and value columns, rename value to the series name
        df = df[['date', 'value']].rename(columns={'value': series_id})
        return df
    else:
        print(f"Error fetching {series_id}: {response.status_code}")
        return None

# Main execution block to test pulling the data
if __name__ == "__main__":
    print("Starting data extraction...")
    all_dfs = []
    
    for name, s_id in SERIES.items():
        print(f"Fetching {name} ({s_id})...")
        df = fetch_fred_series(s_id, FRED_API_KEY)
        if df is not None:
            all_dfs.append(df)
            time.sleep(1) # Polite API spacing
    
    if len(all_dfs) == 2:
        # Merge the datasets together on the date column
        final_df = pd.merge(all_dfs[0], all_dfs[1], on='date', how='outer')
        final_df.to_csv('macro_economic_vectors.csv', index=False)
        print("✅ Success! Created macro_economic_vectors.csv")