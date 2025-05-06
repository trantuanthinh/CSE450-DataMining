class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        return f"{self.year:04}-{self.month:02}-{self.day:02}"

    def is_after(self, other):
        return (self.year, self.month, self.day) > (other.year, other.month, other.day)

    def to_tuple(self):
        return (self.year, self.month, self.day)


date1 = Date(1933, 6, 22)
print(str(date1))
date2 = Date(1933, 9, 17)
print(str(date2))
print(date2.is_after(date1))
print(date1.to_tuple())
