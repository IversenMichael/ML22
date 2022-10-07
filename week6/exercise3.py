import numpy as np
import torch # the torch tensor library

## CREATE SOME DATA
n = 100
x1 = np.random.rand(n)
x2 = np.random.rand(n)
## Grahm schmidt process - ignore
x1 = x1/np.linalg.norm(x1)
x2 = x2/np.linalg.norm(x2)
x2 = x2 - np.dot(x1, x2) * x1 #
x2 = x2/np.linalg.norm(x2)

## CREATE THE DATA MATRIX
a = 4.0
D = np.c_[x1, a*x2]
# CREATE TARGET FUNCTION VECTOR
y = x1 + x2

# MAKE TORCH VARIABLES TO USE
X = torch.from_numpy(D).double()
ty = torch.from_numpy(y).double()
ni = torch.tensor(1./n, dtype=torch.double)

def torch_gd():
    w = torch.tensor([42.0, 2.0], dtype=torch.double)
    lr = torch.tensor(10.0/a, dtype=torch.double)
    epochs = 42
    cost_hist = []
    for i in range(epochs):
        w.requires_grad_() # say i want gradient relative to w in upcoming computation
        cost = torch.mean((X @ w - ty) ** 2)
        cost_hist.append(cost)
        print('epoch {0}: Cost {1}'.format(i, cost))
        cost.backward() # compute gradient cost as function of w
        w = w - lr * w.grad # update w
        w.detach_() # removes w from current computation graph
    print('best w found', w)
torch_gd()