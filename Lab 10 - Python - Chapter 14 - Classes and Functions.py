from datetime import datetime, date, time


class Time:
    def __init__(self, hour, minute, second):
        self.hour = hour
        self.minute = minute
        self.second = second


def subtract_time(t1, t2):
    dt1 = datetime.combine(date.today(), time(t1.hour, t1.minute, t1.second))
    dt2 = datetime.combine(date.today(), time(t2.hour, t2.minute, t2.second))
    delta = dt2 - dt1
    return delta.total_seconds()


def is_after(t1, t2):
    dt1 = datetime.combine(date.today(), time(t1.hour, t1.minute, t1.second))
    dt2 = datetime.combine(date.today(), time(t2.hour, t2.minute, t2.second))
    return dt2 > dt1


class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day


def make_date(year, month, day):
    return Date(year, month, day)


def print_date(date):
    print(f"{date.year:04}-{date.month:02}-{date.day:02}")


def is_after(date1, date2):
    return (date1.year, date1.month, date1.day) > (date2.year, date2.month, date2.day)


t1 = Time(9, 15, 30)
t2 = Time(11, 45, 15)

# Assignment 1
difference = subtract_time(t1, t2)
print(f"Time difference: {difference} seconds")

# Assignment 2
isAfter = is_after(t1, t2)
print(f"Is After: {isAfter}")

# Assignment 3
date1 = make_date(1933, 6, 22)
print_date(date1)
date2 = make_date(1933, 9, 17)
print_date(date2)
print(is_after(date2, date1))
