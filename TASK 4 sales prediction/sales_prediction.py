# ==========================================
# CODEALPHA TASK 4
# SALES PREDICTION USING PYTHON
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# LOAD DATASET
# ==========================================

data = pd.read_csv("Advertising.csv")

# ==========================================
# DISPLAY DATASET
# ==========================================

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET INFORMATION ==========")
print(data.info())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

# ==========================================
# FEATURES AND TARGET
# ==========================================

# If your dataset contains an "Unnamed: 0" column
if "Unnamed: 0" in data.columns:
    data.drop("Unnamed: 0", axis=1, inplace=True)

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# ==========================================
# SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MODEL
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL PERFORMANCE
# ==========================================

print("\n========== MODEL PERFORMANCE ==========")

print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))

print("Mean Squared Error :", mean_squared_error(y_test, y_pred))

print("Root Mean Squared Error :",
      np.sqrt(mean_squared_error(y_test, y_pred)))

print("R2 Score :", r2_score(y_test, y_pred))

# ==========================================
# ACTUAL VS PREDICTED GRAPH
# ==========================================

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.xlabel("Actual Sales")

plt.ylabel("Predicted Sales")

plt.title("Actual vs Predicted Sales")

plt.show()

# ==========================================
# CORRELATION HEATMAP
# ==========================================

plt.figure(figsize=(6, 5))

sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.show()

# ==========================================
# PREDICT NEW SALES
# ==========================================

sample = pd.DataFrame(
    [[230.1, 37.8, 69.2]],
    columns=['TV', 'Radio', 'Newspaper']
)

prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")

print("Predicted Sales =", prediction[0])

# ==========================================
# END OF PROJECT
# ==========================================