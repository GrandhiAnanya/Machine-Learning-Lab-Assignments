import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import tracemalloc


start = time.perf_counter()
tracemalloc.start()
def load_data():
    df=pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )

    return df

def histogram(p1):
    return np.histogram(p1,bins=10)

def mean_var(p1):
    mean=np.mean(p1)
    var=np.var(p1)
    return mean,var



df=load_data()
p1=df["Income"].dropna()

hist,bins=histogram(p1)
print(hist)
print(bins)

plt.hist(p1,bins=10)
plt.xlabel("income")
plt.ylabel("bin value")
plt.show()

mean,var=mean_var(p1)
print(f"Mean : {mean}, Variance : {var}")

current, peak = tracemalloc.get_traced_memory()

print("Current Memory:", current, "bytes")
print("Peak Memory:", peak, "bytes")

tracemalloc.stop()

end = time.perf_counter()
print("Execution Time:", end - start, "seconds")
