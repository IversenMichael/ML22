import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import fetch_california_housing
import os, ssl

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
        getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context


class BaggingRegressor:
    def __init__(self, num_trees, sample_fraction):
        self.num_t = num_trees
        self.p = sample_fraction
        self.trees = []

    def bootstrap_sample(self, X, y, m):
        """ Returns a new data matrix containing m samples with replacement from (X,y)

        Args:
           X: np.array (n, d)  features
           y: np.array (n, ) targets
           m: int, number of bootstrap samples

        returns X_boot, y_boot: np.array shape (m,d), np.array (m, ). Bootstrap samples.
        A data matrix and vector of labels, where each example is a uniform sample from (X,y)
        with replacement (same element can be sampled multiple times).
        """
        X_boot = None
        y_boot = None
        ### YOUR CODE HERE
        n, d = X.shape
        sample_idx = np.random.choice(np.arange(n), size=m)
        X_boot = X[sample_idx, :]
        y_boot = y[sample_idx]
        ### END CODE
        return X_boot, y_boot

    def fit(self, data, targets):
        """ Use bagging to fit multiple regressions trees to the data

        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets

        Appends self.num_t regression trees to self.trees, each trained on a bootstrap sample
        consisting of m = self.p * n examples
        """
        for i in range(self.num_t):
            clf = DecisionTreeRegressor()
            m = (int)(self.p * data.shape[0])
            ### YOUR CODE HERE
            X_boot, y_boot = self.bootstrap_sample(data, targets, m)
            clf.fit(X_boot, y_boot)
            ### END CODE
            self.trees.append(clf)

    def predict(self, X):
        """ Bagging prediction algorithm

        Args
            X: np.array, shape n,d

        returns pred: np.array shape n,  model predictions on X. Average of predictions made by
        regression trees in self.trees
        """
        pred = None
        ### YOUR CODE HERE
        pred = 0
        for tree in self.trees:
            pred += tree.predict(X)
        pred /= len(self.trees)
        ### END CODE
        return pred

    def score(self, X, y):
        """ Compute mean least squares loss of the model

        Args
            X: np.array, shape n,d
            y: np.array, shape n,

        returns out: scalar - mean least squares loss.
        """
        out = None
        ### YOUR CODE HERE
        ### END CODE
        return out


def main():

    # load data
    housing = fetch_california_housing()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(housing.data,
                                                        housing.target,
                                                        test_size=0.2)
    baseline_accuracy = np.mean((y_test - np.mean(y_train)) ** 2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy)

    # Regression tree
    clf = DecisionTreeRegressor()
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    regression_tree_accuracy = np.mean((y_test - predict) ** 2)
    print('Least Squares Cost of RegressionTree:', regression_tree_accuracy)

    # Bagging
    bag = BaggingRegressor(20, 0.7)
    bag.fit(X_train, y_train)
    predict = bag.predict(X_test)
    bagging_accuracy = np.mean((y_test - predict) ** 2)
    print('Least Squares Cost of Bagging:', bagging_accuracy)


if __name__ == '__main__':
    main()