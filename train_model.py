import pandas as pd
from sklearn.model_split import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

print("🚀 INITIALIZING XGBOOST MACHINE LEARNING ENGINE...")
print("--------------------------------------------------")

# 1. Load the processed dataset we created in the last step
df = pd.read_csv("processed_corporate_data.csv")

# 2. Define our Features (Inputs) and Target (Output)
# We will use these key financial ratios as inputs for our model
feature_columns = [
    'currentRatio', 'quickRatio', 'cashRatio', 'daysOfSalesOutstanding',
    'netProfitMargin', 'pretaxProfitMargin', 'grossProfitMargin', 
    'operatingProfitMargin', 'returnOnAssets', 'returnOnEquity', 
    'debtEquityRatio', 'debtRatio'
]

# Ensure we only use columns that are actually present in the file
X = df[feature_columns]
y = df['rating_encoded']  # This is our numerical target (0, 1, 2, etc.)

# 3. Train/Test Split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"📋 Training Set Size: {X_train.shape[0]} corporate records")
print(f"📋 Testing Set Size: {X_test.shape[0]} corporate records\n")

# 4. Initialize and Train the XGBoost Model
print("🧠 Training the decision tree forest (XGBoost)...")
model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)
print("✅ Training complete!")

# 5. Model Validation & Accuracy Check
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("--------------------------------------------------")
print(f"🎯 MODEL ACCURACY SCORE: {accuracy * 100:.2f}%")
print("--------------------------------------------------")
print("\n📝 Detailed Performance Report:")
print(classification_report(y_test, y_pred))