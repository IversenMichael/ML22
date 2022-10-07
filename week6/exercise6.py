import numpy as np
h_in = np.array([[-1, 2, 4]])
d_hout = np.array([[1,2,3]])
print('shapes:', h_in.shape, d_hout.shape)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

def relu_grad(d_hout, h_in):
    J = np.diag([step(x) for x in h_in.flatten()])
    return d_hout @ J

def sigmoid_grad(d_hout, h_in):
    J = np.diag([sigmoid(x)*(1 - sigmoid(x)) for x in h_in.flatten()])
    return d_hout @ J

print('d_hin relu:', relu_grad(d_hout, h_in))
# should be [0, 2, 3]
print('d_hin sigmoid:', sigmoid_grad(d_hout, h_in))
# should be ~ [0.196..., 0.209..., 0.052...]