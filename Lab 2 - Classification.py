import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file = "Dataset of Diabetes.csv"
df = pd.read_csv(file)

# Assignment 1
# 1. Display the basic description of the dataset, including its shape, columns, and data types
print("Shape of dataset:", df.shape)
print("-----")
print("Columns:\n", df.columns)
print("-----")
print("Data types:\n", df.dtypes)
print("-----")
print("Summary statistics:\n", df.describe(include="all"))

print("-----------------")
# 2. Drop the 'ID', 'No_Pation' column
df = df.drop(["ID", "No_Pation"], axis=1)
print("Shape of dataset after drop:", df.shape)
print("-----")
print("Columns after drop:\n", df.columns)

print("-----------------")
# 3. Replace missing values in 'CLASS' with 'P'
print("Before Fill Statistic:", df["CLASS"].isnull().sum())
df["CLASS"] = df["CLASS"].str.strip()
df["CLASS"] = df["CLASS"].fillna("P")
print("After Fill Statistic:", df["CLASS"].isnull().sum())

print("-----------------")
# 4. Boxplot to visualize relationships between 'HbA1c' and 'Gender'
plt.figure(figsize=(8, 5))
sns.boxplot(x="Gender", y="HbA1c", data=df)
plt.title("Boxplot of HbA1c by Gender")
plt.show()

print("-----------------")
# 5. Scatterplot to visualize relationship between 'Age' and 'Urea' with different 'CLASS'
plt.figure(figsize=(8, 5))
sns.scatterplot(x="AGE", y="Urea", hue="CLASS", data=df)
plt.title("Scatterplot of AGE vs Urea by CLASS")
plt.show()

print("-----------------")
# 6. Countplot to visualize 'Gender' distribution with different 'CLASS'
plt.figure(figsize=(8, 5))
sns.countplot(x="Gender", hue="CLASS", data=df)
plt.title("Countplot of Gender by CLASS")
plt.show()

print("-----------------")
# 7. Choose and visualize some relationships
plt.figure()
sns.scatterplot(x="BMI", y="HbA1c", hue="CLASS", data=df)
plt.title("HbA1c & BMI by CLASS")
plt.show()
print("-----")
plt.figure(figsize=(8, 5))
sns.boxplot(x="AGE", y="Gender", hue="CLASS", data=df)
plt.title("AGE & Gender by CLASS")
plt.show()
print("-----")
plt.figure(figsize=(8, 5))
sns.scatterplot(x="Urea", y="Cr", hue="CLASS", data=df)
plt.title("Urea & Cr by CLASS")
plt.show()
