print('Do the python forward and backward pass here')
x1 = 3.0
x2 = 1.0
y = 9.0
w1 = 1.0
w2 = 2.0
w3 = 1.0

def step(x):
    if x >= 0:
        return 1
    else:
        return 0

w1x1 = w1 * x1
w2x2 = w2 * x2
add = w1x1 + w2x2
relu = max(0, add)
nn = w3 * relu
diff = y - nn
e = diff ** 2

de = 1
ddiff = 2 * diff * de
dy = 1 * ddiff
dnn = -1 * ddiff
dw3 = relu * dnn
drelu = w3 * dnn
dadd =  step(add) * drelu
dw1x1 = 1 * dadd
dw2x2 = 1 * dadd
dw1 = x1 * dw1x1
dw2 = x2 * dw2x2
print(f"dw1 = {dw1}")
print(f"dw2 = {dw2}")
print(f"dw3 = {dw3}")
import torch
from torchviz import make_dot # install this package
x1 = torch.tensor([[3.]])
x2 = torch.tensor([[1.]])
y = torch.tensor([9.])
W1 = torch.tensor([[1.]], requires_grad=True)
W2 = torch.tensor([[2.]], requires_grad=True)
W3 = torch.tensor([[1.]], requires_grad=True)
### YOUR CODE HERE - The clamp function may be usefull
cost = (y - W3 * torch.clamp(W1 * x1 + W2 * x2, min=0))**2
cost.backward()
### END CODE
# print the graph - change naming appropriately
make_dot(cost)
print('d_w1', W1.grad)
print('d_w2', W2.grad)
print('d_w3', W3.grad)