import torch
from torch import optim
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

def main():
    dataset = fetch_california_housing()
    X, y = dataset.data, dataset.target
    X = (X - X.mean(axis=0)) / (X.std(axis=0))
    tX = torch.from_numpy(X).float()
    ty = torch.from_numpy(y).float().view(-1, 1)
    net = NN()
    net.fit(tX, ty, hidden_size=16, c=0.01)
    print('pytorch Neural Net least squares score:', net.score(tX, ty).item())

def relu(x):
    return torch.clamp(x, min=0)

class NN():
    def __init__(self):
        self.W1 = None
        self.W2 = None
        self.b1 = None
        self.b2 = None

    def cost(self, X, y, W1, b1, W2, b2, c=0.0):
        """ Compute (Regularized) Least Squares Loss of neural net
        The clamp function may be usefull

          X: torch.tensor shape (n, d) - Data
          y: torch.tensor shape (n, 1) - Targets
          W1: torch.tensor shape (d, h) - weights
          b1: torch.tensor shape (1, h) - bias weight
          W2: torch.tensor shape (h, 1) - weights
          b2: torch.tensor shape (1, 1) - bias weight
          c: ridge regression weight decay parameter

        returns (weight decay) cost tensor
        """
        return torch.mean((relu(relu(X @ W1 + b1) @ W2 + b2) - y) ** 2) + c * (torch.sum(W1 ** 2) + torch.sum(W2 ** 2))

    def fit(self, X, y, hidden_size=32, c=0.01):
        """ GD Learning Algorithm for Ridge Regression with pytorch

         Args:
         X: torch.tensor shape (n, d)
         y: torch.tensor shape (n, 1)
         hidden_size: int
         c: float weight decay parameter (lambda)
        """
        input_dim = X.shape[1]
        W1 = torch.randn(input_dim, hidden_size, requires_grad=True)
        b1 = torch.randn(1, hidden_size, requires_grad=True)
        W2 = torch.randn(hidden_size, 1, requires_grad=True)
        b2 = torch.randn(1, 1, requires_grad=True)
        ### YOUR CODE HERE
        sgd = optim.SGD(params={W1, b1, W2, b2}, lr=0.1)
        for i in range(1000):
            sgd.zero_grad()
            loss = self.cost(X, y, W1, b1, W2, b2, c=c)
            if i % 100 == 0:
                print('epoch:', i, 'least squares (regularized loss)', loss.item())
            loss.backward()
            sgd.step()
        ### END CODE
        self.W1 = W1.clone()
        self.b1 = b1.clone()
        self.W2 = W2.clone()
        self.b2 = b2.clone()

    def score(self, X, y):
        """ Compute least squares cost for model

        Args:
         X: torch.tensor shape (n, d)
         y: torch.tensor shape (n, 1)

        returns least squares score of model on data X with targets y
        """
        return self.cost(X, y, self.W1, self.b1, self.W2, self.b2, c=0.0)

if __name__ == '__main__':
    main()