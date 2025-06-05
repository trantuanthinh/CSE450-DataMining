import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

file = "Student_Performance.csv"
df = pd.read_csv(file)

# Assignment 1
print("Assignment 1: -----------------")
print(df.info())
print("-----")
print(df.describe())
print("-----")
print(df.head())
print("-----")
print("Missing values:\n", df.isnull().sum())
print("-----")
plt.figure(figsize=(8, 5))
sns.histplot(df["Performance Index"], bins=30, kde=True, color="skyblue")
plt.title("Distribution of Performance Index")
plt.xlabel("Performance Index")
plt.ylabel("Frequency")
plt.show()
print("-----")
numeric_df = df.select_dtypes(include=["number"])
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, cmap="Greens", fmt=".2f")
plt.title("Correlation Heatmap (Numerical Features)")
plt.show()

# Assignment 2
print("Assignment 2: -----------------")
df.columns = df.columns.str.strip()
print("Missing values per column:\n", df.isnull().sum())
print("-----")
# Numerical features
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median)
# Categorical features
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
# One-hot encoding (same as Label Encoding)
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
bool_cols = df.select_dtypes(include=["bool"]).columns
for col in bool_cols:
    df[col] = df[col].astype(int)
print(df.head())
print("-----")
print("Missing values after cleaning:\n", df.isnull().sum())

print("-----")
print("Dataset shape before cleaning:", df.shape)
filtered_df = df.copy()
for col in num_cols:
    Q1 = filtered_df[col].quantile(0.25)
    Q3 = filtered_df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = filtered_df[
        (filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)
    ]
print("Dataset shape after cleaning:", filtered_df.shape)

# Assignment 3
print("Assignment 3: -----------------")
num_cols = df.select_dtypes(include=["float64", "int64"]).columns
# Using StandardScaler
scaler = StandardScaler()
df_scaled_std = df.copy()
df_scaled_std[num_cols] = scaler.fit_transform(df_scaled_std[num_cols])
print("Standardized data:\n", df_scaled_std[num_cols].head())
# Using Min-Max Scaler
scaler = MinMaxScaler()
df_scaled_minmax = df.copy()
df_scaled_minmax[num_cols] = scaler.fit_transform(df_scaled_minmax[num_cols])
print("Min-Max Scaled data:\n", df_scaled_minmax[num_cols].head())

# Assignment 4
print("Assignment 4: -----------------")
features = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
target = "Performance Index"
X = df[features]
y = df[target]
#  Add intercept to X (required for statsmodels)
X = sm.add_constant(X)
# 1. Split data train/test (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train:\n", X_train.head())
print("-----")
print("X_test:\n", X_test.head())
print("-----")
print("y_train:\n", y_train.head())
print("-----")
print("y_test:\n", y_test.head())
# 2. Build model OLS based on train set
model = sm.OLS(y_train, X_train).fit()
print("Model summary:\n", model.summary())
# 3. Predict on train/test set
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
# 5. Draw residual plot based on train set
residuals = y_train - y_train_pred
plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_train_pred, y=residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Performance Index: y_train_pred")
plt.ylabel("Residuals: residuals")
plt.title("Residual Plot (Train Set)")
plt.show()
# 6. Draw scatter plot of regression fit (based on train set)
plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_train["Hours Studied"], y=y_train, label="Actual")
sns.lineplot(
    x=X_train["Hours Studied"], y=y_train_pred, color="red", label="Fitted Line"
)
plt.xlabel("Hours Studied")
plt.ylabel("Performance Index")
plt.title("Regression Fit on Hours Studied (Train Set)")
plt.legend()
plt.show()
