import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("car data.csv")

# Display information
print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET INFORMATION ==========")
print(data.info())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Encode all categorical columns
le = LabelEncoder()

categorical_columns = [
    'Car_Name',
    'Fuel_Type',
    'Selling_type',
    'Transmission'
]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Features and target
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n========== MODEL PERFORMANCE ==========")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE :", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score :", r2_score(y_test, y_pred))

# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()