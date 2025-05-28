import statistics
import numpy as np


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
# print(is_prime(7))
# print(is_prime(12))

# Assignment 2
# student_grades = {"Alice": "A", "Bob": "C", "Carol": "B", "David": "D", "Eve": "F"}
# for key, value in student_grades.items():
# print(f"Student {key} has grade {value}")

# Assignment 3
# numbers = [(i, i**2) for i in range(1, 11)]
# mean, median, std_dev = calculate_stats([i[1] for i in numbers])
# print(f"Mean: {mean}, Median: {median}, Standard Deviation: {std_dev}")

# Assignment 4
A = np.arange(100).reshape(10, 10)
determinant = np.linalg.det(A)
print("Determinant of matrix A:", determinant)
