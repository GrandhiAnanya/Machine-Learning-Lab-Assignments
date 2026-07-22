import numpy as np


def dot(a,b):
    total=0
    for i in range(len(a)):
        total=total+(a[i]*b[i])

    return total

def length(a,b):
    len_a=0
    len_b=0
    for i in range(len(a)):
        len_a=len_a+(a[i]**2)
        len_b=len_b+(b[i]**2)

    len_a=len_a**0.5
    len_b=len_b**0.5

    return len_a,len_b

def num_dot(a,b):
    return np.dot(a,b)

def num_length(a,b):
    return np.linalg.norm(a),np.linalg.norm(b)


a=[1,4,8,29,-45]
b=[44,32,21,-2,3]

print("dot product of a and b:",dot(a,b))
print("dot (using numpy) of a and b",num_dot(a,b))
len_a,len_b=length(a,b)
print("length of vector a and b",len_a,"\t",len_b)
len_a2,len_b2=num_length(a,b)
print("length of the vector a and b (using numpy)",len_a2,"\t",len_b2)
