import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score)
import time


def load_data():
    df=pd.read_csv("features.csv")
    return df

    

def Fit( X_train, y_train):
    X_train_fitted = X_train.copy()
    y_train_fitted = y_train.copy()

    return X_train_fitted,y_train_fitted
  

def predict(X_train_fitted,y_train_fitted,X_test,K):
    # PREDICT
    predictions = []
    for test_sample in X_test:
        # -----------------------------------------
        # Calculate distances
        # -----------------------------------------
        distances = []
        for i in range(len(X_train_fitted)):
            distance = 0
            for j in range(X_train_fitted.shape[1]):
                distance += (X_train_fitted[i, j]- test_sample[j]) ** 2
            distance = np.sqrt(distance)
            distances.append((distance, i))
        # -----------------------------------------
        # Sort distances using Bubble Sort
        # -----------------------------------------
        sorted_distances = distances.copy()
        n = len(sorted_distances)
        for i in range(n - 1):
            for j in range(n - i - 1):
                # Distance comparison
                # Index comparison handles equal distances
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


        # -----------------------------------------
        # Select K = 3 nearest neighbors
        # -----------------------------------------
        neighbors = sorted_distances[:K]
        # -----------------------------------------
        # Majority Voting
        # -----------------------------------------
        votes = {}
        for distance, index in neighbors:
            class_label = y_train_fitted[index]
            if class_label not in votes:
                votes[class_label] = 0
            votes[class_label] += 1
        # -----------------------------------------
        # Find class with maximum votes
        # -----------------------------------------
        maximum_votes = max(votes.values())
        tied_classes = []
        for class_label, count in votes.items():
            if count == maximum_votes:
                tied_classes.append(class_label)
        # -----------------------------------------
        # Tie Breaking
        # Choose class of closest neighbor
        # -----------------------------------------
        if len(tied_classes) == 1:
            predicted_class = tied_classes[0]
        else:
            predicted_class = None
            for distance, index in neighbors:
                if y_train_fitted[index] in tied_classes:
                    predicted_class = y_train_fitted[index]
                    break
        # Store prediction
        predictions.append(predicted_class)
    # Display predictions
    print("Predictions:")
    for i in range(len(predictions)):
        print(
            "Test Sample:", i + 1,
            "| Predicted:", predictions[i]
        )
    return predictions

def score(predictions,y_test):
    correct_predictions = 0
    for i in range(len(y_test)):
        if predictions[i] == y_test[i]:
            correct_predictions += 1
    # Calculate accuracy
    accuracy = correct_predictions / len(y_test)
    print("Correct Predictions:",correct_predictions)
    print("Total Test Samples:",len(y_test))
    print("Accuracy:",accuracy)
    print("Accuracy (%):",accuracy * 100)
    return accuracy


times = []

for run in range(10):
    start_time = time.perf_counter()
    df=load_data()
    selected_classes = ["A", "Z"]
    df = df[df["person_id"].isin(selected_classes)]
    df = df.drop(columns='image_name')
    X = df.drop(columns="person_id").to_numpy()
    y = df["person_id"].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X_train, y_train)
    p=neigh.predict(X_test)
    print("test\tpv\tav")
    for i in range(len(p)):
        print(f"{i}\t{p[i]}\t{y_test[i]}")
    print (f"Acuuracy score:{neigh.score(X_test, y_test)}")

    X_train,y_train = Fit(X_train, y_train)
    predictions=predict(X_train,y_train,X_test,3)
    score(predictions,y_test)
    end_time = time.perf_counter()
    
    times.append(end_time - start_time)
    
average_time = sum(times) / len(times)
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