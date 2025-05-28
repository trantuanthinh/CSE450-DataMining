import statistics
import numpy as np
import pandas as pd
import string


def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True


def calculate_stats(numbers):
    mean = statistics.mean(numbers)
    median = statistics.median(numbers)
    std_dev = statistics.stdev(numbers)
    return (mean, median, std_dev)


# Assignment 1
print(is_prime(7))
print(is_prime(12))

# Assignment 2
student_grades = {"Alice": "A", "Bob": "C", "Carol": "B", "David": "D", "Eve": "F"}
for key, value in student_grades.items():
    print(f"Student {key} has grade {value}")

# Assignment 3
numbers = [(i, i**2) for i in range(1, 11)]
mean, median, std_dev = calculate_stats([i[1] for i in numbers])
print(f"Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}")

# Assignment 4
A = np.arange(100).reshape(10, 10)
determinant = np.linalg.det(A)
print("Determinant of matrix A:", determinant)

# Assignment 5
arr = np.random.randint(0, 100, size=10)
normalized_arr = (arr - arr.min()) / (arr.max() - arr.min())
print("Original array:", arr)
print("Normalized array:", normalized_arr)

# Assignment 6
array = np.array([[4, 9, 16], [25, 36, 49]])
print("Square roots of elements in the third column:")
for i in range(array.shape[0]):
    value = array[i, 2]
    sqrt_value = np.sqrt(value)
    print(f"sqrt({value}) = {sqrt_value}")

# Assignment 7
array = np.array([[10, 20, 30], [15, 25, 35], [5, 100, 200]])
mean_rows = np.mean(array, axis=1)
mean_columns = np.mean(array, axis=0)
median_rows = np.median(array, axis=1)
median_columns = np.median(array, axis=0)
var_rows = np.var(array, axis=1)
var_columns = np.var(array, axis=0)
max_var_index = np.argmax(var_rows)
max_var_row = array[max_var_index]
print("Mean along rows:", mean_rows)
print("Mean along columns:", mean_columns)
print("Median along rows:", median_rows)
print("Median along columns:", median_columns)
print("Variance along rows:", var_rows)
print("Variance along columns:", var_columns)
print(f"\nRow with maximum variance (Row {max_var_index}): {max_var_row}")


# Assignment 8
labels = list(string.ascii_uppercase[:10])
values = np.random.randint(1, 101, size=10)
series = pd.Series(data=values, index=labels)
df = series.to_frame(name="Random_Numbers")
print(df)

# Assignment 9
df = pd.read_csv("products-100.csv")
print("Data types of columns:")
print(df.dtypes)
print("\nFirst 10 rows of the DataFrame:")
print(df.head(10))

# Assignment 10
df = pd.read_csv("products-100.csv")
median_price = df["Price"].median()
df["Price"] = df["Price"].fillna(median_price)
mean_price = df["Price"].mean()
std_price = df["Price"].std()
df["price_normalized"] = (df["Price"] - mean_price) / std_price
print(df.head(10))

# Assignment 11
df = pd.read_csv("products-100.csv")
avg_price_by_category = df.groupby("Category")["Price"].mean()
avg_price_df = avg_price_by_category.reset_index()
avg_price_df.columns = ["Category", "Average_Price"]
avg_price_df = avg_price_df.sort_values(by="Average_Price", ascending=False)
print(avg_price_df)
