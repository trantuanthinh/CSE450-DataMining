def hypot(a, b):
    a_squared = a * a
    b_squared = b * b
    sum_squared = a_squared + b_squared
    hypotenuse = sum_squared**0.5
    return hypotenuse


def is_between(a, b, c):
    return a < b < c or a > b > c


def ackermann(m, n):
    if m == 0:
        return n + 1
    if m > 0 and n == 0:
        return ackermann(m - 1, 1)
    if m > 0 and n > 0:
        return ackermann(m - 1, ackermann(m, n - 1))


def GCD(a, b):
    if b == 0:
        return a
    return GCD(b, a % b)


print(hypot(3, 4))
print(is_between(2, 3, 4))
print(is_between(5, 4, 3))
print(is_between(5, 3, 4))
print(ackermann(5, 5))
print(GCD(10, 20))
