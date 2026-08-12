import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc


#chatgpt used
start = time.perf_counter()
tracemalloc.start()
def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def mean_var(p1):
    mean=np.mean(p1)
    var=np.var(p1)
    return mean,var

def hist(df):
    return np.histogram(df, bins=10)


df = load_data()
income = df["Income"].dropna()
mean,var = mean_var(income)
print(f"Mean : {mean}, Variance : {var}")
hist, bin_edges = hist(income)
plt.figure(figsize=(8,5))
plt.bar(bin_edges[:-1],
        hist,
        width=np.diff(bin_edges),
        align='edge',
        edgecolor='black')

plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.show()

current, peak = tracemalloc.get_traced_memory()

print("Current Memory:", current, "bytes")
print("Peak Memory:", peak, "bytes")

tracemalloc.stop()

end = time.perf_counter()
print("Execution Time:", end - start, "seconds")