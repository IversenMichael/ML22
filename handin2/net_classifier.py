import warnings

import numpy as np

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def loss(X, y, params, c=1e-4):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']
    labels = one_in_k_encoding(y, W2.shape[1])
    return - np.sum(labels * np.log(softmax(relu(X @ W1 + b1) @ W2 + b2))) + c * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

def softmax(X):
    """ 
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array). 
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    row_max = np.max(X, axis=1, keepdims=True)
    return np.exp(X - row_max - np.log(np.sum(np.exp(X - row_max), axis=1, keepdims=True)))

def relu(X):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    return np.maximum(0, X)


def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using Xavier/he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


class NetClassifier:
    def __init__(self):
        """ Trivial Init """
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params

        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        return np.argmax(softmax(relu(X @ W1 + b1) @ W2 + b2), axis=1)

    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y (mean 0-1 loss)
        
        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        return np.sum(self.predict(X, params) == y) / X.shape[0]

    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results 
        and then implement the backwards pass using the intermediate stored results
        
        Use the derivative for cost as a function for input to softmax as derived above
        
        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation
        
        Returns 
            cost: scalar - average cross entropy cost with weight decay parameter c
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]
            
        """
        
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        labels = one_in_k_encoding(y, W2.shape[1])

        # ----------- FORWARD PASS ----------- #
        # Cost
        X1_W1 = X @ W1
        X1_W1_b1 = X1_W1 + b1
        X2 = relu(X1_W1_b1)
        X2_W2 = X2 @ W2
        z = X2_W2 + b2
        cost = - np.ma.sum(labels * np.ma.log(softmax(z)))

        # Weight decay
        W1_squared = W1 ** 2
        W2_squared = W2 ** 2
        W1_W2_sum = np.sum(W1_squared) + np.sum(W2_squared)
        weight_decay = c * W1_W2_sum

        # Total loss
        L = cost + weight_decay

        #  ----------- BACKWARD PASS ----------- #
        # Cost
        # d_cost = d_L
        d_z = - labels + softmax(z)
        d_X2_W2 = d_z
        d_b2 = np.sum(d_z, axis=0, keepdims=True)
        d_X2 = d_X2_W2 @ W2.transpose()
        d_W2 = X2.transpose() @ d_X2_W2
        d_X1_W1_b1 = d_X2 * np.heaviside(X1_W1_b1, 0)
        d_X1_W1 = d_X1_W1_b1
        d_b1 = np.sum(d_X1_W1_b1, axis=0, keepdims=True)
        # d_X1 = d_X1_W1 @ W1.transpose()
        d_W1 = X.transpose() @ d_X1_W1

        # Weight decay
        d_W1_weight = c * 2 * W1
        d_W2_weight = c * 2 * W2

        d_w1 = d_W1 + d_W1_weight
        d_w2 = d_W2 + d_W2_weight

        return L, {'d_w1': d_w1, 'd_b1': d_b1, 'd_w2': d_w2, 'd_b2': d_b2}
        
    def fit(self, X_train, y_train, X_val, y_val, params, batch_size=32, lr=0.1, c=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter 
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        n, d = X_train.shape
        hist = {
            'train_loss': [self.cost_grad(X_train, y_train, params=params, c=0)[0]],
            'train_acc': [self.score(X_train, y_train, params=params)],
            'val_loss': [self.cost_grad(X_val, y_val, params=params, c=0)[0]],
            'val_acc': [self.score(X_val, y_val, params=params)]
        }
        best_cost = np.inf
        for _ in range(epochs):
            permutation = np.random.permutation(np.arange(n))
            X_permuted = X_train[permutation, :]
            y_permuted = y_train[permutation]
            for idx_start in np.arange(0, n, batch_size):
                idx_stop = idx_start + batch_size
                X = X_permuted[idx_start:idx_stop, :]
                y = y_permuted[idx_start:idx_stop]
                _, grad = self.cost_grad(X, y, params, c=c)
                for p, dp in [['W1', 'd_w1'], ['b1', 'd_b1'], ['W2', 'd_w2'], ['b2', 'd_b2']]:
                    params[p] -= lr * grad[dp]

            cost_val = self.cost_grad(X_val, y_val, params, c=0)[0]
            if cost_val < best_cost:
                self.params = params
                best_cost = cost_val
            hist['train_acc'].append(self.score(X_train, y_train, params))
            hist['val_acc'].append(self.score(X_val, y_val, params))
            hist['train_loss'].append(self.cost_grad(X_train, y_train, params, c=0)[0])
            hist['val_loss'].append(cost_val)
        return hist


def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        print(dim)
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()


def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')


def test_grad_approx():
    from copy import deepcopy
    np.random.seed(0)
    n = 1000
    d = 10
    hidden_size = 100
    K = 4

    nc = NetClassifier()
    params = get_init_params(d, hidden_size, K)
    X = np.random.randn(n, d)
    Y = np.random.randint(K, size=n)
    cost0, grad = nc.cost_grad(X, Y, params)
    for p, dp in [['W1', 'd_w1'], ['W2', 'd_w2'], ['b1', 'd_b1'], ['b2', 'd_b2']]:
        d_w1 = np.zeros(grad[dp].shape)
        h = 1e-6
        for i in range(d_w1.shape[0]):
            for j in range(d_w1.shape[1]):
                params_h = deepcopy(params)
                params_h[p][i, j] = params_h[p][i, j] + h
                cost_dh, _ = nc.cost_grad(X, Y, params_h)
                d_w1[i, j] = (cost_dh - cost0) / h
        print('Max error = ', np.max(np.abs(d_w1 - grad[dp])))


def main():
    import warnings
    from matplotlib import pyplot as plt
    warnings.simplefilter('error')
    #np.random.seed(0)
    n = 100
    d = 1000
    hidden_size = 10
    K = 4

    nc = NetClassifier()
    params = get_init_params(d, hidden_size, K)
    X = np.random.randn(n, d)
    Y = np.random.randint(K, size=n)
    hist = nc.fit(X, Y, X, Y, params)

    fig, ax = plt.subplots()
    ax.grid()
    ax.plot(np.arange(len(hist['train_loss'])), hist['train_loss'], 'k.')
    ax.plot(np.arange(len(hist['val_loss'])), hist['val_loss'], 'b.')
    plt.show()



if __name__ == '__main__':
    main()