import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

DATASET = "features.csv"
TARGET_COLUMN = "person_id"
K = 3
DISTANCE_METRIC = "euclidean"
# "bubble", "selection", or "insertion"
SORTING_ALGORITHM = "bubble"

def load_data():
    df=pd.read_csv(DATASET)
    return df

def calculate_distance(test_sample, training_sample, metric):
    # Euclidean Distance
    if metric == "euclidean":
        distance = 0
        for i in range(len(test_sample)):
            distance += (
                test_sample[i] - training_sample[i]
            ) ** 2
        distance = np.sqrt(distance)

    # Manhattan Distance
    elif metric == "manhattan":
        distance = 0
        for i in range(len(test_sample)):
            distance += abs(
                test_sample[i] - training_sample[i]
            )
    else:
        raise ValueError(
            "Invalid distance metric. "
            "Choose 'euclidean' or 'manhattan'."
        )
    return distance

def calculate_all_distances(X_train, test_sample):
    distances = []
    for i in range(len(X_train)):
        distance = calculate_distance(
            test_sample,
            X_train[i],
            DISTANCE_METRIC
        )
        distances.append(
            (distance, i)
        )
    return distances


def sort_distances(distances, algorithm):
    sorted_distances = distances.copy()
    if algorithm == "bubble":
        n = len(sorted_distances)
        for i in range(n - 1):
            for j in range(n - i - 1):
                # Compare distance
                # If distances are equal, compare index
                if (
                    sorted_distances[j][0]
                    > sorted_distances[j + 1][0]
                    or
                    (
                        sorted_distances[j][0]
                        == sorted_distances[j + 1][0]
                        and
                        sorted_distances[j][1]
                        > sorted_distances[j + 1][1]
                    )
                ):
                    sorted_distances[j], sorted_distances[j + 1] = \
                        sorted_distances[j + 1], sorted_distances[j]

    elif algorithm == "selection":
        n = len(sorted_distances)
        for i in range(n - 1):
            minimum = i
            for j in range(i + 1, n):
                if (
                    sorted_distances[j][0]
                    < sorted_distances[minimum][0]
                    or
                    (
                        sorted_distances[j][0]
                        == sorted_distances[minimum][0]
                        and
                        sorted_distances[j][1]
                        < sorted_distances[minimum][1]
                    )
                ):

                    minimum = j

            sorted_distances[i], sorted_distances[minimum] = \
                sorted_distances[minimum], sorted_distances[i]

    elif algorithm == "insertion":
        for i in range(1, len(sorted_distances)):
            current = sorted_distances[i]
            j = i - 1
            while j >= 0 and (
                sorted_distances[j][0] > current[0]
                or
                (
                    sorted_distances[j][0] == current[0]
                    and
                    sorted_distances[j][1] > current[1]
                )
            ):

                sorted_distances[j + 1] = \
                    sorted_distances[j]

                j -= 1

            sorted_distances[j + 1] = current

    else:

        raise ValueError(
            "Invalid sorting algorithm. "
            "Choose 'bubble', 'selection', or 'insertion'."
        )

    return sorted_distances


def identify_neighbors(sorted_distances,k,y_train):
    # Select first k nearest samples
    neighbors = sorted_distances[:k]
    return neighbors

def classify(neighbors, y_train):
    # Store classes of nearest neighbors
    neighbor_classes = []
    for distance, index in neighbors:
        neighbor_classes.append(y_train[index])
    # Count occurrences of each class
    class_counts = Counter( neighbor_classes)
    maximum_votes = max(class_counts.values())
    # Classes having maximum votes
    tied_classes = []
    for class_label, count in class_counts.items():
        if count == maximum_votes:
         tied_classes.append(class_label)
    if len(tied_classes) == 1:
        predicted_class = tied_classes[0]
    else:
        best_distance = float("inf")
        predicted_class = None
        for distance, index in neighbors:
            if y_train[index] in tied_classes:
                if distance < best_distance:
                    best_distance = distance
                    predicted_class = y_train[index]
    return predicted_class, class_counts

def classify_weighted(neighbors, y_train):
    # Store total weighted votes for each class
    class_weights = {}
    for distance, index in neighbors:
        class_label = y_train[index]
        # Avoid division by zero
        epsilon = 1e-10
        # Calculate weight
        weight = 1 / (distance + epsilon)
        # Add weight to the corresponding class
        if class_label not in class_weights:
            class_weights[class_label] = 0
        class_weights[class_label] += weight
    # Find maximum weighted vote
    maximum_weight = max(class_weights.values())
    # Find classes having maximum weight
    tied_classes = []
    for class_label, weight in class_weights.items():
        if weight == maximum_weight:
            tied_classes.append(class_label)
    # Tie-breaking
    if len(tied_classes) == 1:
        predicted_class = tied_classes[0]
    else:
        # If weighted votes are tied,
        # choose the class of the closest neighbor
        predicted_class = None
        closest_distance = float("inf")
        for distance, index in neighbors:
            if y_train[index] in tied_classes:
                if distance < closest_distance:
                    closest_distance = distance
                    predicted_class = y_train[index]
    return predicted_class, class_weights


df=load_data()
selected_classes = ["A", "Z"]
df = df[df["person_id"].isin(selected_classes)]

df = df.drop(columns=["image_name"])

X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# Convert to NumPy
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

# ============================================================
# K VALUE EXPERIMENT
# ============================================================

# Different K values to test
K_VALUES = [1, 3, 5, 7]

# Store accuracy results
knn_accuracies = []
wknn_accuracies = []
library_accuracies = []


# ============================================================
# TEST DIFFERENT K VALUES
# ============================================================

for K in K_VALUES:
    predictions = []
    weighted_predictions = []
    # Predict every test sample
    for i in range(len(X_test)):
        test_sample = X_test[i]
        # ----------------------------------------
        # Calculate distances
        # ----------------------------------------
        distances = calculate_all_distances(X_train,test_sample)
        # ----------------------------------------
        # Sort distances
        # ----------------------------------------
        sorted_distances = sort_distances(distances,SORTING_ALGORITHM)
        # ----------------------------------------
        # Identify K neighbors
        # ----------------------------------------
        neighbors = identify_neighbors(sorted_distances,K,y_train)
        # ----------------------------------------
        # Normal KNN
        # ----------------------------------------
        predicted_class, class_counts = classify(neighbors,y_train)
        predictions.append(predicted_class)

        # ----------------------------------------
        # Weighted KNN
        # ----------------------------------------
        weighted_class, class_weights = classify_weighted(neighbors,y_train)
        weighted_predictions.append(weighted_class)


    # ========================================================
    # ACCURACY OF OUR NORMAL KNN
    # ========================================================

    correct = 0

    for i in range(len(y_test)):

        if predictions[i] == y_test[i]:

            correct += 1

    knn_accuracy = (
        correct / len(y_test)
    ) * 100

    knn_accuracies.append(
        knn_accuracy
    )


    # ========================================================
    # ACCURACY OF OUR WEIGHTED KNN
    # ========================================================

    correct = 0

    for i in range(len(y_test)):

        if weighted_predictions[i] == y_test[i]:

            correct += 1

    wknn_accuracy = (
        correct / len(y_test)
    ) * 100

    wknn_accuracies.append(
        wknn_accuracy
    )


    # ========================================================
    # LIBRARY BASED KNN
    # ========================================================

    neigh = KNeighborsClassifier(
        n_neighbors=K,
        metric=DISTANCE_METRIC
    )

    neigh.fit(
        X_train,
        y_train
    )
    p=neigh.predict(X_test)

    library_accuracy = (
        neigh.score(
            X_test,
            y_test
        )
    ) * 100

    library_accuracies.append(
        library_accuracy
    )

# ============================================================
# DISPLAY ALL ACCURACY RESULTS
# ============================================================

print("\n========================================")
print("ACCURACY COMPARISON")
print("========================================")

for i in range(len(K_VALUES)):

    print(
        "K =", K_VALUES[i],
        "| KNN =", knn_accuracies[i], "%",
        "| WKNN =", wknn_accuracies[i], "%",
        "| Library KNN =", library_accuracies[i], "%"
    )

# ============================================================
# GRAPH 1
# WEIGHTED KNN VS LIBRARY KNN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    K_VALUES,
    wknn_accuracies,
    marker="o",
    label="Weighted KNN"
)

plt.plot(
    K_VALUES,
    library_accuracies,
    marker="o",
    label="Library KNN"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")

plt.title(
    "Weighted KNN vs Library KNN"
)

plt.xticks(K_VALUES)

plt.legend()

plt.grid(True)

plt.show()

# ============================================================
# GRAPH 2
# NORMAL KNN VS LIBRARY KNN
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    K_VALUES,
    knn_accuracies,
    marker="o",
    label="Implemented KNN"
)

plt.plot(
    K_VALUES,
    library_accuracies,
    marker="o",
    label="Library KNN"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy (%)")

plt.title(
    "Implemented KNN vs Library KNN"
)

plt.xticks(K_VALUES)

plt.legend()

plt.grid(True)

plt.show()
