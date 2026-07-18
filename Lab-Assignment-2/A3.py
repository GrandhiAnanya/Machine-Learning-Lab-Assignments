import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

def execution_time(function,data):
    total=0
    for i in range(10):
        start=time.perf_counter()
        function(data)
        end=time.perf_counter()
        total+=(end-start)
    average_time=total/10

    return average_time

def load_data():
    df = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="IRCTC Stock Price"
    )
    return df

def calculate_stats(df):
    mean=np.mean(df["Price"])
    var=np.var(df["Price"])
    return mean,var

def mean_og(price):
    total=0

    for value in price:
        total+=value
    mean=total/len(price)

    return mean

def var_og(price):
    mean=mean_og(price)
    total=0

    for value in price:
        total+=(value-mean)**2
    var=total/len(price)
    return var

def wednesday_mean(df):
    wednesday = df[df["Day"] == "Wed"]
    mean = np.mean(wednesday["Price"])

    return mean

def april_mean(df):
    april = df[df["Month"] == "Apr"]
    mean = np.mean(april["Price"])

    return mean

def probability_loss(df):
    loss = df[df["Chg%"]<0]
    prob=len(loss)/len(df)

    return prob

def probability_wed(df):
    wed = df[df["Day"] == "Wed"]
    profit = wed[wed["Chg%"] > 0]
    prob = len(profit)/len(wed)

    return prob

def prob_condprofit_wed(df):
    wed = df[df["Day"] == "Wed"]
    profit = wed[wed["Chg%"] > 0]
    prob = len(profit)/len(wed)

    return prob

def scatter_plot(df):
    plt.figure (figsize=(10,5))
    plt.scatter (df["Day"], df["Chg%"])
    plt.title("Scatter plot of Day vs Chg%")
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.grid(True)
    plt.show()




df=load_data()

mean, var = calculate_stats(df)
price=df["Price"]

print("Mean Price using numpy:", mean)
print("Mean price:", mean_og(price))
print("Variance using numpy:", var)
print("Variance:", var_og(price))

print("\nMean Error:", abs(mean - mean_og(price)))
print("Variance Error:", abs(var - var_og(price)))


numpy_mean_time = execution_time(np.mean, price)
mean_time = execution_time(mean_og, price)

numpy_var_time = execution_time(np.var, price)
var_time = execution_time(var_og, price)

print("\nExecution Time Comparison")
print("NumPy Mean Time      :", numpy_mean_time)
print("Mean Time         :", mean_time)
print("NumPy Variance Time  :", numpy_var_time)
print("Variance Time     :", var_time)

wed_avg=wednesday_mean(df)
print("\nOverall Mean :", mean)
print("Wednesday Mean:", wed_avg)
if wed_avg > mean:
    print("Wednesday average is higher than the overall average.")
elif wed_avg < mean:
    print("Wednesday average is lower than the overall average.")
else:
    print("Both averages are equal.")

apr_avg=april_mean(df)
print("\nOverall Mean :", mean)
print("April Mean:", apr_avg)
if apr_avg > mean:
    print("April average is higher than the overall average.")
elif apr_avg < mean:
    print("April average is lower than the overall average.")
else:
    print("Both averages are equal.")

loss_prob=probability_loss(df)
print(f"\nProbability of Loss : {loss_prob*100:.4f}%")

profit_wed=probability_wed(df)
print(f"\nProbability of Profit on Wednesday: {profit_wed*100:.2f}%")

cond_profit=prob_condprofit_wed(df)
print(f"\nConditional probability of Profit given that it is Wednesday: {cond_profit*100:.2f}%")

scatter_plot(df)