import math
import turtle
import numpy as np

screen = turtle.Screen()
t = turtle.Turtle()


# Assignment 3
def draw_parallelogram(width, height, angle):
    t.forward(width)
    t.left(180 - angle)
    t.forward(height)
    t.left(angle)
    t.forward(width)
    t.left(180 - angle)
    t.forward(height)
    t.hideturtle()
    screen.mainloop()


def draw_rectangle(width, height):
    draw_parallelogram(width, height, 90)


def draw_rhombus(width, height, angle):
    draw_parallelogram(width, height, angle)


def draw_pie(number, edge):
    acute_angle = 360 / number
    bottom_angle = (180 - acute_angle) / 2
    bottom_edge = edge * (
        np.sin(np.radians(acute_angle)) / np.sin(np.radians(bottom_angle))
    )

    t.right(acute_angle / 2)
    for i in range(1, number + 1):
        draw_isosceles_triangle(edge, acute_angle, bottom_edge, bottom_angle)
        t.penup()
        t.home()
        t.pendown()
        t.left(acute_angle * (i - 1) + acute_angle / 2)
    t.hideturtle()
    screen.mainloop()


def draw_isosceles_triangle(edge, acute_angle, bottom_edge, bottom_angle):
    t.forward(edge)
    t.left(90 + (acute_angle / 2))
    t.forward(bottom_edge)
    t.left(90 + (90 - bottom_angle))
    t.forward(edge)


def draw_flower(number, radius, angle):
    for _ in range(number):
        draw_petal(radius, angle)
        t.left(360 / number)
    t.hideturtle()
    screen.mainloop()


def draw_petal(radius, angle):
    for _ in range(2):
        draw_arc(radius, angle)
        t.left(180 - angle)


def draw_arc(radius, angle):
    arc_length = 2 * math.pi * radius * abs(angle) / 360
    n = int(arc_length / 5) + 1
    step_length = arc_length / n
    step_angle = angle / n

    t.left(step_angle / 2)
    for _ in range(n):
        t.forward(step_length)
        t.left(step_angle)
    t.right(step_angle / 2)


t.pensize(5)
t.speed(0)

# Assignment 1
draw_rectangle(200, 100)

# Assignment 2
draw_rhombus(200, 100, 60)

# Assignment 4
draw_pie(100, 100)

# Assignment 5
draw_flower(4, 100, 60)
