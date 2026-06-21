import pandas as pd
import numpy as np

print("🚀 INITIALIZING 3-BUCKET CREDIT RATINGS MAPPER...")
print("--------------------------------------------------")

# 1. Load data
df = pd.read_csv("corporate_rating.csv")

# Clean Date and extract tracking year
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Year'] = df['Date'].dt.year.fillna(2015)

# 2. Hardcoded Macro Bucketing Function (Claude's Strategy)
def bin_credit_rating(rating_str):
    if not isinstance(rating_str, str):
        return 0
    rating = rating_str.strip().upper()
    
    # Bucket 2: High Grade (AAA down to A-)
    if rating in ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-']:
        return 2
    # Bucket 1: Medium Grade (BBB family)
    elif rating in ['BBB+', 'BBB', 'BBB-']:
        return 1
    # Bucket 0: Speculative Grade / Low (BB and everything below)
    else:
        return 0

# Apply the custom macro mapping to your target column
df['rating_encoded'] = df['Rating'].apply(bin_credit_rating)

print("🔢 Target Classes compressed into 3 Strategic Operational Buckets:")
print("   🔹 Label 2 -> High Grade Investment (AAA to A-)")
print("   🔹 Label 1 -> Medium Grade Investment (BBB Family)")
print("   🔹 Label 0 -> Speculative Grade / Junk (BB and below)")

print("\n📊 New Balanced Category Class Breakdown:")
print(df['rating_encoded'].value_counts())

# 3. Clean up the Missing Ratios
financial_columns = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 
    'operatingProfitMargin', 'returnOnAssets', 'returnOnEquity', 
    'debtEquityRatio', 'debtRatio'
]
valid_features = [col for col in financial_columns if col in df.columns]

for col in valid_features:
    df[col] = df[col].fillna(df[col].median())

# Generate Sector Benchmarks
for col in valid_features:
    sector_median = df.groupby('Sector')[col].transform('median')
    df[f'{col}_sector_ratio'] = df[col] / (sector_median + 1e-5)

df['macro_interest_proxy'] = np.where(df['Year'] > 2014, 2.5, 0.5)
df['profit_to_debt_leverage'] = df['netProfitMargin'] / (df['debtRatio'] + 1e-5)

# 4. Export high-powered macro training base
output_file = "processed_corporate_data.csv"
df.to_csv(output_file, index=False)
print("--------------------------------------------------")
print(f"🥇 SUCCESS! Consolidated data matrix written to: '{output_file}'")