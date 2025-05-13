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


print_right("Monty")
print_right("Python's")
print_right("Flying Circus")
triangle("L", 5)
rectangle("H", 5, 4)
