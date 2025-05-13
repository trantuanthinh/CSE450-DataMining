import time


now = time.time()


def is_triangle(a, b, c):
    return a + b > c and a + c > b and b + c > a


second_in_1_day = 60 * 60 * 24
days_since_epoch = int(now // second_in_1_day)

seconds_today = int(now % second_in_1_day)

hours = seconds_today // 3600
minutes = (seconds_today % 3600) // 60
seconds = seconds_today % 60

print(f"Days since 1970-01-01: {days_since_epoch}")
print(f"Current time (UTC): {hours:02d}:{minutes:02d}:{seconds:02d}")
print(is_triangle(3, 4, 5))

# n=3, s=0 -> recurse(n=3-1=2,s=3+0=3)
# n=2, s=3 -> recurse(n=2-1=1,s=2+3=5)
# n=1, s=5 -> recurse(n=1-1=0,s=1+5=6)
# n=0, s=6 -> print(s=6)
