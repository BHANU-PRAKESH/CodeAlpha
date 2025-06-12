# car_price_prediction.py

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
# Step 1: Load the data
df = pd.read_csv(r"C:\Users\HP\Downloads\archive (5)\car data.csv")
# Step 2: Preprocessing
# Drop car name (not useful for ML)
df.drop('Car_Name', axis=1, inplace=True)
# Convert 'Year' to car age
df['Car_Age'] = 2025 - df['Year']
df.drop('Year', axis=1, inplace=True)
# Encode categorical variables
df = pd.get_dummies(df, drop_first=True)
# Step 3: Define Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']
# Step 4: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
# Step 5: Train model
model = RandomForestRegressor(n_estimators=100, random_state=50)
model.fit(X_train, y_train)
# Step 6: Evaluate model
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
# Step 7: Save the model
joblib.dump(model, 'car_price_model.pkl')
print("Model saved as car_price_model.pkl")
# Optional: Visualize Feature Importance
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)
plt.figure(figsize=(10,6))
plt.title('Feature Importance')
plt.barh(range(len(indices)), importances[indices], align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()