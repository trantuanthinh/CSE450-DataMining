import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-01-23/english_education.csv"
df = pd.read_csv(url)

# Assignment 1
print(df.head())
print("-----")
print(df.shape)  # (rows, columns)
print("-----")
print(df.columns)
print("-----")
print(df.dtypes)
print("-----")
print(df.info())

# Assignment 2
# Numerical Features
# Using Pandas
print(df.describe())
df["population_2011"].hist(bins=200)
plt.title("Histogram of population_2011 using Pandas")
plt.xlabel("Value")
plt.ylabel("Frequency")
print("Histogram of population_2011 using Pandas")
plt.show()
print("-----")
# Using Seaborn
sns.histplot(df["population_2011"], bins=200)
plt.title("Histogram of population_2011 using Seaborn")
plt.xlabel("Value")
plt.ylabel("Frequency")
print("Histogram of population_2011 using Seaborn")
plt.show()

print("-----------------")
# Categorical Features
# Using Pandas
print("Categorical Feature - ", df["size_flag"].value_counts())
df["size_flag"].value_counts().plot(kind="bar")
plt.title("Bar plot of size_flag using Pandas")
plt.xlabel("Size Flag")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
print("Bar plot of size_flag using Pandas")
plt.show()

print("-----")
# Using Seaborn
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="size_flag")
plt.title("Bar plot of size_flag using Seaborn")
plt.xlabel("Size Flag")
plt.ylabel("Count")
plt.grid(axis="y")
plt.tight_layout()
print("Bar plot of size_flag using Seaborn")
plt.show()

# Assignment 3
# Pair 1: key_stage_2_attainment_school_year_2007_to_2008 & key_stage_4_attainment_school_year_2012_to_2013 by job_density_flag
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="key_stage_2_attainment_school_year_2007_to_2008",
    y="key_stage_4_attainment_school_year_2012_to_2013",
    hue="job_density_flag",
)
plt.title(
    "key_stage_2_attainment_school_year_2007_to_2008 & key_stage_4_attainment_school_year_2012_to_2013 by job_density_flag"
)
plt.grid(True)
plt.tight_layout()
print(
    "key_stage_2_attainment_school_year_2007_to_2008 & key_stage_4_attainment_school_year_2012_to_2013 by job_density_flag"
)
plt.show()

print("-----")
# Pair 2: activity_at_age_19_employment_with_earnings_above_0 & activity_at_age_19_employment_with_earnings_above_10_000 by education_score
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df,
    x="activity_at_age_19_employment_with_earnings_above_0",
    y="activity_at_age_19_employment_with_earnings_above_10_000",
    hue="education_score",
)
plt.title(
    "activity_at_age_19_employment_with_earnings_above_0 & activity_at_age_19_employment_with_earnings_above_10_000 by education_score"
)
plt.grid(True)
plt.tight_layout()
print(
    "activity_at_age_19_employment_with_earnings_above_0 & activity_at_age_19_employment_with_earnings_above_10_000 by education_score"
)
plt.show()

print("-----")
# Pair 3: level_2_at_age_18 & level_3_at_age_18 by income_flag
plt.figure(figsize=(6, 4))
sns.scatterplot(
    data=df, x="level_2_at_age_18", y="level_3_at_age_18", hue="income_flag"
)
plt.title("level_2_at_age_18 & level_3_at_age_18 by income_flag")
plt.grid(True)
plt.tight_layout()
print("level_2_at_age_18 & level_3_at_age_18 by income_flag")
plt.show()

print("-----")
# Boxplot
plt.figure(figsize=(6, 4))
sns.boxplot(
    data=df, x="level_2_at_age_18", y="level_3_at_age_18", hue="university_flag"
)
plt.title("level_2_at_age_18 & level_3_at_age_18 by university_flag")
plt.grid(True)
plt.tight_layout()
print("level_2_at_age_18 & level_3_at_age_18 by university_flag")
plt.show()

# Assignment 4
print("Missing Values by Column:")
print(df.isnull().sum())
print("-----")
missing_percent = df.isnull().mean() * 100
print("Percentage of Missing Values by Column:")
print(missing_percent[missing_percent > 0])
print("-----")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"{col}: {len(outliers)} outliers")

# Assignment 5
print("Missing values per column:\n", df.isnull().sum())
df.columns = df.columns.str.strip()
for col in df.select_dtypes(include=["float64", "int64"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include=["object", "category"]).columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode().iloc[0])
print("-----")
print("Missing values after cleaning:\n", df.isnull().sum())

# Assignment 6
print("Original shape:", df.shape)
print("-----")
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
print("Shape after outlier removal:", df.shape)

# Assignment 7
df["total_qualification_levels"] = (
    df["highest_level_qualification_achieved_by_age_22_less_than_level_1"]
    + df["highest_level_qualification_achieved_by_age_22_level_1_to_level_2"]
    + df["highest_level_qualification_achieved_by_age_22_level_3_to_level_5"]
)
print(df[["total_qualification_levels"]].head())

# Assignment 8
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df[numerical_cols] = df[numerical_cols].apply(lambda x: np.log1p(x - x.min() + 1))
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
print(df.head())
