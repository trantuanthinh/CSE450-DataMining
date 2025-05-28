import math


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return "({0}, {1})".format(self.x, self.y)


class Line:
    def __init__(self, point1, point2):
        self.point1 = point1
        self.point2 = point2

    def __str__(self):
        return "{0} to {1}".format(self.point1, self.point2)

    # Exercise 1
    def __eq__(self, other):
        return self.point1.__eq__(other.point1) and self.point2.__eq__(other.point2)

    # Exercise 2
    def midpoint(self):
        mid_x = (self.point1.x + self.point2.x) / 2
        mid_y = (self.point1.y + self.point2.y) / 2
        return Point(mid_x, mid_y)


class Rectangle:
    def __init__(self, bottom_line, top_line, right_line, left_line):
        self.bottom_line = bottom_line
        self.top_line = top_line
        self.right_line = right_line
        self.left_line = left_line

    # Exercise 3
    def midpoint(self):
        mid_point_width = self.bottom_line.midpoint()
        mid_point_height = self.right_line.midpoint()
        return Point(mid_point_width.x, mid_point_height.y)

    def make_lines(self):
        bottom_left = self.bottom_line.point1
        top_left = self.left_line.point2
        top_right = self.top_line.point1
        bottom_right = self.right_line.point1

        return [
            Line(bottom_left, top_left),
            Line(top_left, top_right),
            Line(top_right, bottom_right),
            Line(bottom_right, bottom_left),
        ]

    # Exercise 4
    def make_cross(self):
        sides = self.make_lines()
        midpoints = [line.midpoint() for line in sides]
        return [
            Line(midpoints[0], midpoints[2]),  # left to right
            Line(midpoints[1], midpoints[3]),  # top to bottom
        ]


# Exercise 5
class Circle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius

    def __str__(self):
        return "Center: {0}, Radius: {1}".format(self.center, self.radius)

    def circumference(self):
        return 2 * math.pi * self.radius

    def area(self):
        return math.pi * self.radius**2


# Assignment 1, 2, 3, 4, 5
# Example usage
p1 = Point(0, 0)
p2 = Point(1, 0)
p3 = Point(1, 1)
p4 = Point(0, 1)
print("Point 1:", p1)
print("Point 2:", p2)
print("Point 3:", p3)
print("Point 4:", p4)
print("--")

bottom_line = Line(p1, p2)
top_line = Line(p3, p4)
right_line = Line(p2, p3)
left_line = Line(p1, p4)
print("Bottom line:", bottom_line)
print("Top line:", top_line)
print("Right line:", right_line)
print("Left line:", left_line)
print("--")

print("Midpoint of bottom line:", bottom_line.midpoint())
print("Midpoint of top line:", top_line.midpoint())
print("Midpoint of right line:", right_line.midpoint())
print("Midpoint of left line:", left_line.midpoint())
print("--")

rect = Rectangle(bottom_line, top_line, right_line, left_line)
print("Midpoint of rectangle:", rect.midpoint())
print("--")

cross_lines = rect.make_cross()
for line in cross_lines:
    print("Cross line:", line)
print("--")

circle = Circle(Point(0, 0), 5)
print(circle)
print("Circumference:", circle.circumference())
print("Area:", circle.area())
print("--")
