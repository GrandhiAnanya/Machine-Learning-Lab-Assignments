import numpy as np
import pandas as pd


def rank_d(X):
    rank=np.linalg.matrix_rank(X)
    return rank

def cost(X,y):
    c = np.linalg.pinv(X) @ y
    return c


df = pd.read_excel("Lab Session Data.xlsx" ,
                 sheet_name="Purchase data")

X = df[[
    "Candies (#)",
    "Mangoes (Kg)",
    "Milk Packets (#)"
]]

y = df["Payment (Rs)"]

X = X.to_numpy()
y = y.to_numpy()

print(X)
print(y)

rank=rank_d(X)
print("rank of the feature matrix",rank)

c=cost(X,y)
print("cost of each product is:\n")
print(c)



