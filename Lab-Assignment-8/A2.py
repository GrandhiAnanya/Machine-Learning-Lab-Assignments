import numpy as np
import matplotlib.pyplot as plt

def summation_unit(x, w, b):
    total = 0
    for i, weight in zip(x, w):
        total += i * weight
    total += b
    return total


def step_function(sum):
    return 1 if sum >= 0 else 0


def error(y, t):
    return t - y


def calculate_sse(errors):
    total = 0
    for i in errors:
        total += i ** 2
    return total


w0 = 10
w1 = 0.2
w2 = -0.75
n = 0.05

A = [0, 0, 1, 1]
B = [0, 1, 0, 1]
target = [0, 0, 0, 1]
training_err = []

for epoch in range(1, 1001):
    train = []
    for k, l, t in zip(A, B, target):
        y = summation_unit([k, l], [w1, w2], w0)
        pred = step_function(y)
        err = error(pred, t)
        train.append(err)
        delta_w0 = -n * (pred - t) * 1
        delta_w1 = -n * (pred - t) * k
        delta_w2 = -n * (pred - t) * l
        w0 = w0 + delta_w0
        w1 = w1 + delta_w1
        w2 = w2 + delta_w2
    epoch_sse = calculate_sse(train)
    training_err.append(epoch_sse)
    if epoch_sse <= 0.002:
        print("Converged at epoch:", epoch)
        break


print("Final weights:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)

plt.plot(range(1, len(training_err) + 1), training_err)
plt.xlabel("Epochs")
plt.ylabel("Sum-Square Error")
plt.title("Epochs vs Sum-Square Error")
plt.show()