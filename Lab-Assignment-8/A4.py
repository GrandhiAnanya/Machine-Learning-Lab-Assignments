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
n_all = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

A = [0, 0, 1, 1]
B = [0, 1, 0, 1]
target = [0, 0, 0, 1]
converg_epoch = []

for n in n_all:
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
        if epoch_sse <= 0.002:
            print(f"Converged at epoch:{epoch} leanring rate:{n}")
            converg_epoch.append(epoch)
            break


plt.plot(converg_epoch, n_all)
plt.xlabel("leanrning rate")
plt.ylabel("epoch it converged")
plt.show()