def f(x):
    return x * x

def f_(x):
    return 2 * x

def gradient_descent(x, r, time):
    for i in range(time):
        x = x - r * f_(x)
        print(f"x: {x} - f(x):{f(x)}")


r = 0.1
x = 10
time = 10
gradient_descent(x, r, time)
