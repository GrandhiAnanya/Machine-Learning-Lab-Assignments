import math

def minkowski_distance(p1,p2,o):
    dist=0
    for i in range(len(p1)):
        dist+=abs(p1[i]-p2[i])**o

    dist=pow(dist,1/o)

    return dist

def euclidean(p1,p2):
    return minkowski_distance(p1,p2,2)

def manhattan(p1,p2):
    return minkowski_distance(p1,p2,1)


p1=[4,6,3]
p2=[7,8,9]
e=euclidean(p1,p2)
m=manhattan(p1,p2)

print("Euclidean distance: ",e)
print("\n Manhattan distance: ",m)





    