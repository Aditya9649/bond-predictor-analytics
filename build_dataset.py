import pandas as pd
import numpy as np

print("🚀 INITIALIZING CORPORATE BOND PREDICTOR DATA ENGINE...")
print("---------------------------------------------------------")

# 1. Setup paths for your raw sources
bond_data_path = "raw_trace_bonds.csv"

try:
    # Load your market transaction dataset
    bonds_df = pd.read_csv(bond_data_path)
    bonds_df['TRADE_DATE'] = pd.to_datetime(bonds_df['TRADE_DATE'])
    print(f"✅ Loaded {bonds_df.shape[0]} transaction rows from TRACE.")
    
    print("\n🌐 FETCHING REAL MACRO DATASETS DIRECTLY FROM FRED...")
    # Downloading public economic benchmarks dynamically from the Federal Reserve repository
    fed_rate_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
    treasury_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
    cpi_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL"
    
    fed_df = pd.read_csv(fed_rate_url, parse_dates=['DATE']).rename(columns={'FEDFUNDS': 'FED_FUNDS_RATE'})
    t10_df = pd.read_csv(treasury_url, parse_dates=['DATE']).rename(columns={'GS10': 'TREASURY_10Y_YIELD'})
    cpi_df = pd.read_csv(cpi_url, parse_dates=['DATE']).rename(columns={'CPIAUCSL': 'CPI_INFLATION'})
    
    # 2. Merge the dataframes into an enterprise feature layout
    print("🔄 Merging datasets on trading date matrix lines...")
    # Forward fill handles weekends/holidays where economic indicators don't change
    macro_df = fed_df.merge(t10_df, on='DATE', how='outer').merge(cpi_df, on='DATE', how='outer')
    macro_df = macro_df.sort_values('DATE').ffill()
    
    # Join macro data vectors straight into your market transactions
    final_dataset = pd.merge_asof(
        bonds_df.sort_values('TRADE_DATE'), 
        macro_df, 
        left_on='TRADE_DATE', 
        right_on='DATE', 
        direction='backward'
    )
    
    # 3. Export the combined machine learning feature base
    output_filename = "engineered_bond_features.csv"
    final_dataset.to_csv(output_filename, index=False)
    print(f"\n🥇 SUCCESS! Combined dataset written to: '{output_filename}'")
    print(f"📊 Final feature vector shape: {final_dataset.shape[0]} rows x {final_dataset.shape[1]} columns.")

except FileNotFoundError:
    print(f"❌ Error: Cannot find your bond trade database at: '{bond_data_path}'")
    print("👉 Action: Make sure your bond dataset is saved in this folder with that exact filename.")
except Exception as e:
    print(f"⚠️ Network/Parsing issue occurred: {e}")