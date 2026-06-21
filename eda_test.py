import pandas as pd

print("⏳ CONNECTING TO DATA MATRIX...")

try:
    # Load the corporate dataset
    df = pd.read_csv("corporate_rating.csv")
    
    print("\n✅ CONNECTION SUCCESSFUL!")
    print("---------------------------------------")
    print(f"📊 Dataset size: {df.shape[0]} companies | {df.shape[1]} metrics")
    
    # Check what target grades are in column 1
    print("\n🔍 Snapshot of target credit ratings in the file:")
    print(df['Rating'].value_counts().head(5))

except Exception as e:
    print(f"❌ Something went wrong: {e}")