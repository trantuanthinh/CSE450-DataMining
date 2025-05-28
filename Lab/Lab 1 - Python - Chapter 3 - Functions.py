def print_right(text):
    print(" " * (40 - len(text)) + text)


def triangle(text, n):
    for i in range(1, n + 1):
        print(text * i)


def rectangle(text, width, height):
    for i in range(1, height + 1):
        print(text * width)


def bottle_verse(n):
    print(n, "bottles of beer on the wall,", n, "bottles of beer.")
    print("Take one down, pass it around,", n - 1, "bottles of beer on the wall.")


# Assignment 1
print_right("Monty")
print_right("Python's")
print_right("Flying Circus")

# Assignment 2
triangle("L", 5)

# Assignment 3
rectangle("H", 5, 4)

# Assignment 4
for i in range(99, 0, -1):
    bottle_verse(i)
    print()
