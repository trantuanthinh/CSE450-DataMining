import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the historical spending data
url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-01-23/english_education.csv"
# url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2024/2024-02-13/historical_spending.csv"
df = pd.read_csv(url)

# # Assignment 1
# print(df.head())
# print(df.shape)  # (rows, columns)
# print(df.columns)
# print(df.dtypes)
# print(df.info())

# Assignment 2
# Numerical Features
# # Using Pandas
# print(df.describe())
# df["population_2011"].hist(bins=200)
# plt.title("Histogram of population_2011")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()
# # Using Seaborn
# sns.histplot(df["population_2011"], bins=200)
# plt.title("Histogram of population_2011")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# plt.show()

# Categorical Features
df["size_flag"].value_counts()
