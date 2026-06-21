# ======================================
# CODEALPHA TASK 2
# UNEMPLOYMENT ANALYSIS WITH PYTHON
# ======================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Display first 5 rows
print("\n========== FIRST 5 ROWS ==========")
print(data.head())

# Dataset information
print("\n========== DATASET INFORMATION ==========")
print(data.info())

# Missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Statistical summary
print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

# Average unemployment by region
region_unemployment = data.groupby(
    "Region"
)["Estimated Unemployment Rate (%)"].mean()

print("\n========== AVERAGE UNEMPLOYMENT RATE ==========")
print(region_unemployment)

# ------------------------------
# BAR GRAPH
# ------------------------------
plt.figure(figsize=(12,6))
region_unemployment.sort_values().plot(kind='bar')
plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=90)
plt.show()

# ------------------------------
# HEATMAP
# ------------------------------
plt.figure(figsize=(8,5))

sns.heatmap(
    data.select_dtypes(include=['float64', 'int64']).corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.show()

# ------------------------------
# REGION WISE ANALYSIS
# ------------------------------
plt.figure(figsize=(12,6))

sns.boxplot(
    x='Region',
    y='Estimated Unemployment Rate (%)',
    data=data
)

plt.xticks(rotation=90)
plt.title("Region-wise Unemployment Distribution")
plt.show()

print("\n========== ANALYSIS COMPLETED ==========")