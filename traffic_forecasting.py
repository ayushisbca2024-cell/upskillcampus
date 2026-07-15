import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ======================================
# LOAD DATASET
# ======================================

current_folder = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_folder, "train_aWnotuB.csv")

df = pd.read_csv(csv_path)

print("===================================")
print("SMART CITY TRAFFIC FORECASTING")
print("===================================\n")

print("First 5 Rows:\n")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ======================================
# FEATURE ENGINEERING
# ======================================

print("\nConverting DateTime...")

df["DateTime"] = pd.to_datetime(df["DateTime"])

df["Year"] = df["DateTime"].dt.year
df["Month"] = df["DateTime"].dt.month
df["Day"] = df["DateTime"].dt.day
df["Hour"] = df["DateTime"].dt.hour
df["DayOfWeek"] = df["DateTime"].dt.dayofweek
df["Weekday"] = df["DateTime"].dt.day_name()

df["Weekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)

print("Feature Engineering Completed!")

# ======================================
# EXPLORATORY DATA ANALYSIS
# ======================================

print("\nGenerating graphs...")

# Graph 1
junction_avg = df.groupby("Junction")["Vehicles"].mean()

plt.figure(figsize=(6,4))
junction_avg.plot(kind="bar")
plt.title("Average Traffic at Each Junction")
plt.xlabel("Junction")
plt.ylabel("Average Vehicles")
plt.tight_layout()
plt.savefig("graph1_junction.png")
plt.close()

# Graph 2
hourly_avg = df.groupby("Hour")["Vehicles"].mean()

plt.figure(figsize=(10,5))
plt.plot(hourly_avg.index, hourly_avg.values, marker="o")
plt.title("Average Traffic by Hour")
plt.xlabel("Hour")
plt.ylabel("Average Vehicles")
plt.grid(True)
plt.tight_layout()
plt.savefig("graph2_hourly.png")
plt.close()

# Graph 3
monthly_avg = df.groupby("Month")["Vehicles"].mean()

plt.figure(figsize=(8,5))
monthly_avg.plot(kind="bar")
plt.title("Average Traffic by Month")
plt.xlabel("Month")
plt.ylabel("Average Vehicles")
plt.tight_layout()
plt.savefig("graph3_monthly.png")
plt.close()

# Graph 4
plt.figure(figsize=(8,5))
plt.hist(df["Vehicles"], bins=30)
plt.title("Distribution of Vehicle Count")
plt.xlabel("Vehicles")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("graph4_distribution.png")
plt.close()

# Graph 5
numeric_df = df.select_dtypes(include=["number"])

plt.figure(figsize=(10,6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("graph5_heatmap.png")
plt.close()

print("Graphs saved successfully!")

# ======================================
# MACHINE LEARNING MODEL
# ======================================

print("\nPreparing data for Machine Learning...")

X = df[["Junction", "Year", "Month", "Day", "Hour", "DayOfWeek", "Weekend"]]
y = df["Vehicles"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Model Trained Successfully!")

print("Making Predictions...")

predictions = model.predict(X_test)

print("Predictions Completed!")

# ======================================
# MODEL EVALUATION
# ======================================

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ======================================
# ACTUAL VS PREDICTED
# ======================================

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nSample Predictions:\n")
print(comparison.head(10))

comparison.to_csv("traffic_predictions.csv", index=False)

print("\nPredictions saved as traffic_predictions.csv")

print("\nProject Completed Successfully!")