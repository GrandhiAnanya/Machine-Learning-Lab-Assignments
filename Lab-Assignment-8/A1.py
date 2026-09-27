import numpy as np

def summation_unit(x, w, b):
    total = 0
    for i, weight in zip(x, w):
        total += i * weight
    total += b
    return total

def step_function(sum):
    return 1 if sum>=0 else 0

def bipolar_step_function(sum):
    return 1 if sum>=0 else -1

def sigmoid_function(sum):
    return (1/(1+np.exp(-1*sum)))

def tanh_function(sum):
    e_y = np.exp(sum)
    e_y_minus = np.exp(-1*sum)

    num=e_y-e_y_minus
    den=e_y+e_y_minus

    return(num/den)

def relu_function(sum):
    return sum if sum>0 else 0

def leaky_relu_function(sum):
    return sum if sum>0 else 0.01*sum

def error(y,t):
      return t-y


#and gate
A=[0,0,1,1]
B=[0,1,0,1]
target=[0,0,0,1]
w=[1,1]
b=-1.5

output_step = []
output_bipolar = []
output_sigmoid = []
output_tanh = []
output_relu = []
output_leaky_relu = []

error_step = []
error_bipolar = []
error_sigmoid = []
error_tanh = []
error_relu = []
error_leaky_relu = []

for i, j, t in zip(A, B, target):

    y = summation_unit([i, j], w, b)

    
    step = step_function(y)
    bipolar = bipolar_step_function(y)
    sigmoid = sigmoid_function(y)
    tanh = tanh_function(y)
    relu = relu_function(y)
    leaky_relu = leaky_relu_function(y)

   
    output_step.append(step)
    output_bipolar.append(bipolar)
    output_sigmoid.append(sigmoid)
    output_tanh.append(tanh)
    output_relu.append(relu)
    output_leaky_relu.append(leaky_relu)

    
    error_step.append(error(step, t))
    error_bipolar.append(error(bipolar, t))
    error_sigmoid.append(error(sigmoid, t))
    error_tanh.append(error(tanh, t))
    error_relu.append(error(relu, t))
    error_leaky_relu.append(error(leaky_relu, t))


print("Step Output:", output_step)
print("Step Error:", error_step)

print("Bipolar Step Output:", output_bipolar)
print("Bipolar Step Error:", error_bipolar)

print("Sigmoid Output:", output_sigmoid)
print("Sigmoid Error:", error_sigmoid)

print("Tanh Output:", output_tanh)
print("Tanh Error:", error_tanh)

print("ReLU Output:", output_relu)
print("ReLU Error:", error_relu)

print("Leaky ReLU Output:", output_leaky_relu)
print("Leaky ReLU Error:", error_leaky_relu)


