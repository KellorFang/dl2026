import matplotlib.pyplot as plt

filepath = "lr.csv"


w0 = 0
w1 = 1
x = []
y = []
l = []

with open(filepath) as f:
    for line in f:
        x.append(float(line.strip().split(",")[0]))
        y.append(float(line.strip().split(",")[1]))



def f(w0, w1, xi, yi):
    return (1 / 2) * (w0 + w1 * xi - yi) ** 2

def df0(w0, w1, xi, yi):
    return w1 * xi + w0 - yi

def df1(w0, w1, xi, yi):
    return xi * (w1 * xi + w0 - yi)

def j(w0, w1, xi, yi):
    N = len(x)
    return (1 / N) * sum(f(w0, w1, xi, yi) for xi, yi in zip(x,y))

def grad_desc(w0, w1, xi, yi, lr, threshold):
    prev_loss = float('inf')
    loss = j(w0, w1, xi, yi)
    while abs(prev_loss - loss) > threshold:
        prev_loss = loss
        w0 = w0 - lr * sum(df0(w0, w1, xi, yi) for xi, yi in zip(x,y))
        w1 = w1 - lr * sum(df1(w0, w1, xi, yi) for xi, yi in zip(x,y))
        loss = j(w0, w1, xi, yi)
        l.append(loss)
        print(f"w0: {w0}, w1: {w1}, loss: {loss}")
    return w0, w1


w0, w1 = grad_desc(w0, w1, x, y, 0.00001, 0.0000001)

plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, [w0 + w1 * xi for xi in x], color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig("l1.png")
plt.show()

plt.plot(range(len(l)), l)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.show()

