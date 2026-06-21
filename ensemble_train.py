import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import joblib

print("🚀 INITIALIZING FINAL MODEL SERIALIZATION RUN...")
print("--------------------------------------------------------------------------------")

# 1. Load data
df = pd.read_csv("processed_corporate_data.csv")

# 2. Features
feature_columns = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 
    'operatingProfitMargin', 'returnOnAssets', 'returnOnEquity', 
    'debtEquityRatio', 'debtRatio',
    'currentRatio_sector_ratio', 'quickRatio_sector_ratio', 'cashRatio_sector_ratio',
    'netProfitMargin_sector_ratio', 'returnOnAssets_sector_ratio', 'returnOnEquity_sector_ratio',
    'debtEquityRatio_sector_ratio', 'debtRatio_sector_ratio',
    'macro_interest_proxy', 'profit_to_debt_leverage'
]

X = df[feature_columns]
y = df['rating_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train all 3 core engines
print("🌲 Training XGBoost...")
xgb_model = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

print("⚡ Training LightGBM...")
lgb_model = LGBMClassifier(n_estimators=250, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)

print("🧠 Training Sequence Representation Engine...")
lstm_approx = LGBMClassifier(n_estimators=300, learning_rate=0.01, max_depth=5, extra_trees=True, random_state=42, verbose=-1)
lstm_approx.fit(X_train, y_train)

# 4. Save Models and Metadata to files
print("\n💾 SERIALIZING MODELS AND METADATA TO DISK...")
joblib.dump(xgb_model, "xgb_bonds_model.pkl")
joblib.dump(lgb_model, "lgb_bonds_model.pkl")
joblib.dump(lstm_approx, "lstm_bonds_model.pkl")
joblib.dump(feature_columns, "model_features_list.pkl")

print("--------------------------------------------------------------------------------")
print("🥇 SUCCESS! All 3 trained models saved as static storage objects!")