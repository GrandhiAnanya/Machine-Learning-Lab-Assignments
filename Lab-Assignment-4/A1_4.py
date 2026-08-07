import numpy as np

#chatgpt used
def minkowski_distance(point1, point2, p):
        point1 = np.array(point1)
        point2 = np.array(point2)

        if len(point1) != len(point2):
            raise ValueError("Both points must have the same number of dimensions.")

        distance = np.sum(np.abs(point1 - point2) ** p) ** (1 / p)

        return distance

point1 = [2, 3, 4]
point2 = [5, 7, 1]

print("Manhattan Distance:", minkowski_distance(point1, point2, 1))

print("Euclidean Distance:", minkowski_distance(point1, point2, 2))

print("Minkowski Distance (p=3):", minkowski_distance(point1, point2, 3))