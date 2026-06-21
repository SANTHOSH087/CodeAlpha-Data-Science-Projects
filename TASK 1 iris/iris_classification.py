# ==========================================
# CODEALPHA TASK 1 - IRIS FLOWER CLASSIFICATION
# ==========================================

# Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# LOAD DATASET
# ==========================================
data = pd.read_csv("Iris.csv")

print("\n========== FIRST 5 ROWS ==========")
print(data.head())

print("\n========== DATASET INFORMATION ==========")
print(data.info())

print("\n========== DATASET SHAPE ==========")
print(data.shape)

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

print("\n========== SPECIES COUNT ==========")
print(data["Species"].value_counts())

# ==========================================
# FEATURE AND TARGET SELECTION
# ==========================================
X = data.iloc[:, 1:5]
y = data.iloc[:, 5]

# ==========================================
# SPLIT DATA INTO TRAINING AND TESTING
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# CREATE MODEL
# ==========================================
model = DecisionTreeClassifier(random_state=42)

# ==========================================
# TRAIN MODEL
# ==========================================
model.fit(X_train, y_train)

# ==========================================
# PREDICTION
# ==========================================
y_pred = model.predict(X_test)

# ==========================================
# ACCURACY
# ==========================================
accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL ACCURACY ==========")
print("Accuracy =", accuracy)

# ==========================================
# CONFUSION MATRIX
# ==========================================
print("\n========== CONFUSION MATRIX ==========")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# CLASSIFICATION REPORT
# ==========================================
print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(y_test, y_pred))

# ==========================================
# PREDICT NEW FLOWER
# ==========================================
sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

print("\n========== NEW FLOWER PREDICTION ==========")
print("Predicted Species :", prediction[0])

# ==========================================
# VISUALIZATION
# ==========================================
species_count = data["Species"].value_counts()

plt.figure(figsize=(7,5))
species_count.plot(kind="bar")

plt.xlabel("Species")
plt.ylabel("Count")
plt.title("Distribution of Iris Species")

plt.show()

# ==========================================
# END OF PROJECT
# ==========================================