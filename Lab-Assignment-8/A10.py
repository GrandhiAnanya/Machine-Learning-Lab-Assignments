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


target = [
    [0, 1],
    [0, 1],
    [0, 1],
    [1, 0]
]

n = 0.05


v11 = 0.1
v12 = 0.1
v21 = 0.1
v22 = 0.1

w11 = 0.1
w12 = 0.1
w21 = 0.1
w22 = 0.1

epochs = 1000

for epoch in range(1, epochs + 1):

    errors = []

    for a, b, t in zip(A, B, target):

        net_h1 = a * v11 + b * v21
        h1 = sigmoid_function(net_h1)

        net_h2 = a * v12 + b * v22
        h2 = sigmoid_function(net_h2)

       
        net_o1 = h1 * w11 + h2 * w21
        o1 = sigmoid_function(net_o1)

        net_o2 = h1 * w12 + h2 * w22
        o2 = sigmoid_function(net_o2)


        error_o1 = t[0] - o1
        error_o2 = t[1] - o2

        errors.append(error_o1)
        errors.append(error_o2)

       

        delta_o1 = error_o1 * o1 * (1 - o1)
        delta_o2 = error_o2 * o2 * (1 - o2)

    

        delta_h1 = h1 * (1 - h1) * (
            delta_o1 * w11 + delta_o2 * w12
        )

        delta_h2 = h2 * (1 - h2) * (
            delta_o1 * w21 + delta_o2 * w22
        )

    

        delta_w11 = n * delta_o1 * h1
        delta_w21 = n * delta_o1 * h2

        delta_w12 = n * delta_o2 * h1
        delta_w22 = n * delta_o2 * h2

        w11 = w11 + delta_w11
        w21 = w21 + delta_w21

        w12 = w12 + delta_w12
        w22 = w22 + delta_w22

  

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



print("\nFinal weights:")

print("v11 =", v11)
print("v12 =", v12)
print("v21 =", v21)
print("v22 =", v22)

print("w11 =", w11)
print("w12 =", w12)
print("w21 =", w21)
print("w22 =", w22)