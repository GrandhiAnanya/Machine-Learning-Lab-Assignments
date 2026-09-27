import numpy as np
import pandas as pd

def load_data():
    df=pd.read_csv('customer_data.csv')
    return df

def summation_unit(x, w, b):
    total = 0

    for i, weight in zip(x, w):
        total += i * weight

    total += b
    return total

def sigmoid_function(sum):
    return (1/(1+np.exp(-1*sum)))

def error(y, t):
    return t - y

def calculate_sse(errors):
    total = 0
    for i in errors:
        total += i ** 2
    return total



df=load_data()
X = df[["Candies", "Mangoes", "Milk_Packets", "Payment"]].values
target = df["High_Value"].values
w0 = 0.1
w1 = 0.1
w2 = 0.1
w3 = 0.1
w4 = 0.1
w = [w1, w2, w3, w4]
n = 0.1
epochs = 1000

for epoch in range(1, epochs + 1):
    total_error = []
    for x, t in zip(X, target):
        y = summation_unit(x, w, w0)
        pred = sigmoid_function(y)
        err = error(pred, t)
        total_error.append(err)
        delta_w0 = -n * (pred - t) * 1
        delta_w1 = -n * (pred - t) * x[0]
        delta_w2 = -n * (pred - t) * x[1]
        delta_w3 = -n * (pred - t) * x[2]
        delta_w4 = -n * (pred - t) * x[3]
        w0 = w0 + delta_w0
        w1 = w1 + delta_w1
        w2 = w2 + delta_w2
        w3 = w3 + delta_w3
        w4 = w4 + delta_w4
        w = [w1, w2, w3, w4]
    sse=calculate_sse(total_error)



print("Final weights:")
print("w0 =", w0)
print("w1 =", w1)
print("w2 =", w2)
print("w3 =", w3)
print("w4 =", w4)

predictions=[]
for x, t in zip(X, target):
    y = summation_unit(x, w, w0)
    pred = sigmoid_function(y)
    if pred >= 0.5:
        predictions.append(1)
    else:
        predictions.append(0)
    


for i in range(len(predictions)):
    print(f"predicted value: {predictions[i]} target value:{target[i]}")