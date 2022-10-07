import torch
from torch import optim
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
import os, ssl

def main():

    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    print('*' * 5, 'Load and Prepare Data', '*' * 5)
    dataset = fetch_california_housing()
    # print('dataset', dataset)
    X, y = dataset.data, dataset.target
    X = (X - X.mean(axis=0)) / (X.std(axis=0))
    # print('data stats', X.shape, X.mean(axis=0), X.std(axis=0))
    ridge = linear_model.Ridge(alpha=0.1, fit_intercept=True)
    ridge.fit(X, y)
    # print(ridge.coef_, ridge.intercept_)
    print('\n', '*' * 5, 'Test Sklearn Ridge Regression for Comparison', '*' * 5)
    print('Ridge Regression Score:', ((ridge.predict(X) - y) ** 2).mean())

    print('\n', '*' * 5, 'Make data to torch tensors', '*' * 5)
    tX = torch.from_numpy(X).float()
    ty = torch.from_numpy(y).float().view(-1, 1)

    print('\n', '*' * 5, 'Run Torch Linear Regression Gradient Descent', '*' * 5)

    tlr = LR()
    tlr.fit(tX, ty, 0.1)
    print('pytorch Linear Regression least squares score:', tlr.score(tX, ty).item())

class LR():
    def __init__(self):
        self.w = None
        self.b = None

    def cost(self, X, y, w, b, c=0):
        """ Compute Regularized Least Squares Loss

          X: torch.tensor shape (n, d) - Data
          y: torch.tensor shape (n, 1) - Targets
          w: torch.tensor shape (d, 1) - weights
          b: torch.tensor shape (1, 1) - bias weight
          c: scalar, weight decay parameter

          returns (regularized) cost tensor
        """
        loss = torch.mean((X @ w + b - y) ** 2)
        reg_loss = torch.sum(w ** 2)
        return loss + c * reg_loss

    def fit(self, X, y, c=0):
        """ GD Learning Algorithm for Ridge Regression with pytorch

        Args:
         X: torch.tensor shape (n, d)
         y: torch.tensor shape (n, 1)
         c: ridge regression weight decay parameter (lambda)
        """
        w = torch.zeros(X.shape[1], 1, requires_grad=True)
        b = torch.zeros(1, 1, requires_grad=True)
        sgd = optim.SGD(params={w, b}, lr=0.1)
        for i in range(100):
            sgd.zero_grad()
            loss = self.cost(X, y, w, b, c=c)
            if i % 10 == 0:
                print('epoch:', i, 'least squares (regularized loss)', loss.item())
            loss.backward()
            sgd.step()
        self.w = w.clone()
        self.b = b.clone()

    def score(self, X, y):
        """ Compute least squares cost for model

        Args:
         X: torch.tensor shape (n, d)
         y: torch.tensor shape (n, 1)

        returns least squares score of model on data X with targets y
        """
        score = self.cost(X, y, self.w, self.b, c=0)
        return score

if __name__ == '__main__':
    main()