import numpy as np

def sigmoid_function(sum):
    return 1 / (1 + np.exp(-sum))

def calculate_sse(errors):
    total = 0
    for i in errors:
        total += i ** 2
    return total



A = [0, 0, 1, 1]
B = [0, 1, 0, 1]
target = [0, 1, 1, 0]


n = 0.05


v11 = 0.1
v12 = 0.1
v21 = 0.1
v22 = 0.1

w1 = 0.1
w2 = 0.1

epochs = 1000

for epoch in range(1, epochs + 1):
    errors = []
    for a, b, t in zip(A, B, target):
        net_h1 = a * v11 + b * v21
        h1 = sigmoid_function(net_h1)
        net_h2 = a * v12 + b * v22
        h2 = sigmoid_function(net_h2)
        net_o = h1 * w1 + h2 * w2
        o = sigmoid_function(net_o)
        error = t - o
        errors.append(error)

    
        delta_o = error * o * (1 - o)
        delta_h1 = h1 * (1 - h1) * (delta_o * w1)
        delta_h2 = h2 * (1 - h2) * (delta_o * w2)
        delta_w1 = n * delta_o * h1
        delta_w2 = n * delta_o * h2
        w1 = w1 + delta_w1
        w2 = w2 + delta_w2
        delta_v11 = n * delta_h1 * a
        delta_v21 = n * delta_h1 * b
        delta_v12 = n * delta_h2 * a
        delta_v22 = n * delta_h2 * b
        v11 = v11 + delta_v11
        v21 = v21 + delta_v21
        v12 = v12 + delta_v12
        v22 = v22 + delta_v22
    sse = calculate_sse(errors)
    print("Epoch:", epoch, "SSE:", sse)
    if sse <= 0.002:
        print("Converged at epoch:", epoch)
        break