import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from collections import Counter
from sklearn.model_selection import train_test_split 
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
import time

DATASET = "features.csv"
TARGET_COLUMN = "person_id"
K = 3
DISTANCE_METRIC = "euclidean"
# "bubble", "selection", or "insertion"
SORTING_ALGORITHM = "bubble"

def load_data():
    df=pd.read_csv(DATASET)
    return df


def encoding(df):
    # Separate features and target
    X = df.drop("person_id", axis=1)
    y = df["person_id"]

    # Find categorical columns
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Find numerical columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # Identify binary and multi-category columns
    binary_cols = []
    onehot_cols = []

    for col in categorical_cols:
        if X[col].nunique() == 2:
            binary_cols.append(col)
        else:
            onehot_cols.append(col)

    # -------------------------
    # Label Encoding
    # -------------------------

    label_encoders = {}

    for col in binary_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le

    # Encode target variable if categorical
    if y.dtype == "object" or y.dtype.name == "category" or y.dtype == "bool":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y)

    # -------------------------
    # One-Hot Encoding
    # -------------------------

    if onehot_cols:
        encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )

        encoded_data = encoder.fit_transform(X[onehot_cols])

        encoded_columns = encoder.get_feature_names_out(onehot_cols)

        encoded_df = pd.DataFrame(
            encoded_data,
            columns=encoded_columns,
            index=X.index
        )

        # Remove original categorical columns
        X = X.drop(columns=onehot_cols)

        # Add one-hot encoded columns
        X = pd.concat([X, encoded_df], axis=1)

def imputation(df):
    # Numerical columns
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns

    # Categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns


    # Mean Imputation
    # Suitable for numerical data without significant outliers
    for col in numerical_cols:
        if abs(df[col].skew()) < 1:
            df[col] = df[col].fillna(df[col].mean())


    # Median Imputation
    # Suitable for numerical data with outliers
    for col in numerical_cols:
        if abs(df[col].skew()) >= 1:
            df[col] = df[col].fillna(df[col].median())


    # Mode Imputation
    # Suitable for categorical data
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])


    '''print("Missing values after imputation:")
    print(df.isnull().sum())'''

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


times = []

for run in range(10):
    start_time = time.perf_counter()
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

    # Store predictions
    predictions = []
    # Predict every test sample
    for i in range(len(X_test)):
        # Select current test sample
        test_sample = X_test[i]
        distances = calculate_all_distances(X_train,test_sample)

        sorted_distances = sort_distances( distances,SORTING_ALGORITHM)

        neighbors = identify_neighbors(sorted_distances,K,y_train)

        predicted_class, class_counts = classify_weighted(neighbors,y_train)
        # Store prediction
        predictions.append(predicted_class)

    end_time = time.perf_counter()
        
    times.append(end_time - start_time)
        
average_time = sum(times) / len(times)


print("\n========================================")
print("KNN PREDICTION RESULTS")
print("========================================")

for i in range(len(X_test)):
    print("Test Sample:", i + 1,"| Actual:", y_test[i],"| Predicted:", predictions[i])

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="binary",
    pos_label="A"
)

recall = recall_score(
    y_test,
    predictions,
    average="binary",
    pos_label="A"
)

f1 = f1_score(
    y_test,
    predictions,
    average="binary",
    pos_label="A"
)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)
print("Avg Time :", average_time, "seconds")