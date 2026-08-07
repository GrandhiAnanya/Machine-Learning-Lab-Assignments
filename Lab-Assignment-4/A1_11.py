import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import time
import tracemalloc


#chatgpt used
start = time.perf_counter()
tracemalloc.start()

def load_data():
    df = pd.read_excel(
        "Lab Session Data.xlsx",
        sheet_name="marketing_campaign"
    )
    return df


def preprocess(df):

    # Drop unnecessary columns
    df.drop(columns=["ID", "Z_CostContact", "Z_Revenue"], inplace=True)

    # Handle missing values
    df["Income"] = df["Income"].fillna(df["Income"].mean())

    # Label Encoding
    le = LabelEncoder()
    df["Education"] = le.fit_transform(df["Education"])

    # One-Hot Encoding
    df = pd.get_dummies(df, columns=["Marital_Status"], dtype=int)

    # Convert Date into numerical values
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"])
    df["Customer_Year"] = df["Dt_Customer"].dt.year
    df["Customer_Month"] = df["Dt_Customer"].dt.month
    df["Customer_Day"] = df["Dt_Customer"].dt.day
    df.drop(columns=["Dt_Customer"], inplace=True)

    return df


# -----------------------------
# Euclidean Distance Function
# -----------------------------
def euclidean_distance(point1, point2):
    distance = 0

    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2

    return distance ** 0.5


# -----------------------------
# Load and Preprocess
# -----------------------------
df = load_data()
df = preprocess(df)

# Store original dataframe for plotting
plot_df = df.copy()

# -----------------------------
# Normalize Features
# -----------------------------
scaler = MinMaxScaler()
X = scaler.fit_transform(df)

K = 3

# -----------------------------
# Initial Centroids
# -----------------------------
np.random.seed(42)

random_indices = np.random.choice(len(X), K, replace=False)
centroids = X[random_indices]

# -----------------------------
# K-Means Algorithm
# -----------------------------
max_iterations = 100

for iteration in range(max_iterations):

    clusters = []

    # Assign each point to nearest centroid
    for point in X:

        distances = []

        for centroid in centroids:
            distances.append(euclidean_distance(point, centroid))

        clusters.append(np.argmin(distances))

    clusters = np.array(clusters)

    # Compute New Centroids
    new_centroids = []

    for i in range(K):

        cluster_points = X[clusters == i]

        if len(cluster_points) == 0:
            new_centroids.append(centroids[i])
        else:
            new_centroids.append(cluster_points.mean(axis=0))

    new_centroids = np.array(new_centroids)

    # Stop if centroids do not change
    if np.allclose(centroids, new_centroids):
        print(f"Converged after {iteration + 1} iterations.")
        break

    centroids = new_centroids


# -----------------------------
# Results
# -----------------------------
print("\nFinal Centroids (Normalized):\n")
print(centroids)

print("\nCluster Labels:\n")
print(clusters)

# -----------------------------
# Plot (Original Values)
# -----------------------------
plt.figure(figsize=(8,6))

colors = ['red', 'blue', 'green']

for i in range(K):

    plt.scatter(
        plot_df.loc[clusters == i, "Income"],
        plot_df.loc[clusters == i, "MntWines"],
        color=colors[i],
        label=f'Cluster {i+1}'
    )

# Convert centroids back to original scale for plotting
original_centroids = scaler.inverse_transform(centroids)

plt.scatter(
    original_centroids[:, df.columns.get_loc("Income")],
    original_centroids[:, df.columns.get_loc("MntWines")],
    color="black",
    marker="X",
    s=250,
    label="Centroids"
)

plt.xlabel("Income")
plt.ylabel("MntWines")
plt.title("K-Means Clustering")
plt.legend()
plt.grid(True)
plt.show()

current, peak = tracemalloc.get_traced_memory()

print("Current Memory:", current, "bytes")
print("Peak Memory:", peak, "bytes")

tracemalloc.stop()

end = time.perf_counter()
print("Execution Time:", end - start, "seconds")