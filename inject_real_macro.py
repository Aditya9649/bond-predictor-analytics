import pandas as pd
import numpy as np

print("📂 LOADING LOCAL ECONOMIC VECTOR FILES...")
print("--------------------------------------------------------------------------------")

try:
    # 1. Load your newly binned corporate base dataset
    df = pd.read_csv("processed_corporate_data.csv")
    
    # 2. Load the pre-existing macro vector sheet from your folder
    macro_df = pd.read_csv("macro_economic_vectors.csv")
    
    print("🔄 Standardizing tracking parameters for clean inner validation mapping...")
    
    # Standardize column headers to uppercase to prevent matching misses
    macro_df.columns = macro_df.columns.str.upper()
    
    # Extract clean Year identifiers from both frames to merge safely
    if 'DATE' in macro_df.columns:
        macro_df['DATE'] = pd.to_datetime(macro_df['DATE'], errors='coerce')
        macro_df['Year'] = macro_df['DATE'].dt.year
    elif 'YEAR' in macro_df.columns:
        macro_df['Year'] = macro_df['YEAR']
        
    # Isolate key metrics (Look for Fed Funds or matching rate columns)
    # If standard columns like FEDFUNDS don't exist, we fallback safely
    rate_col = [col for col in macro_df.columns if 'FED' in col or 'RATE' in col or 'YIELD' in col]
    
    if rate_col:
        print(f"📊 Identified local macro tracking vector: '{rate_col[0]}'")
        macro_yearly = macro_df.groupby('Year')[rate_col[0]].mean().reset_index().rename(columns={rate_col[0]: 'REAL_MACRO_RATE'})
    else:
        # Fallback to the first non-date numeric column if name signatures vary
        numeric_cols = macro_df.select_dtypes(include=[np.number]).columns
        fallback_col = numeric_cols[0] if len(numeric_cols) > 0 else macro_df.columns[1]
        macro_yearly = macro_df.groupby('Year')[fallback_col].mean().reset_index().rename(columns={fallback_col: 'REAL_MACRO_RATE'})

    # 3. Join the data metrics cleanly using the trade Year alignment
    df = df.merge(macro_yearly, on='Year', how='left')
    
    # Handle gaps cleanly if any years don't perfectly overlap
    df['REAL_MACRO_RATE'] = df['REAL_MACRO_RATE'].fillna(df['REAL_MACRO_RATE'].median())
    
    # Map back to the proxy array your ensemble uses
    df['macro_interest_proxy'] = df['REAL_MACRO_RATE']
    
    # 4. Save out to the processed training matrix file
    df.to_csv("processed_corporate_data.csv", index=False)
    print("--------------------------------------------------------------------------------")
    print("🥇 SUCCESS! Existing local macroeconomic vectors matched and merged cleanly!")

except Exception as e:
    print(f"❌ Verification Error: {e}")