import numpy as np

def relu(x):
    return np.maximum(0, x)

class NN():
    def __init__(self, input_dim, hidden_size):
        output_size = 1
        self.W1 = np.random.rand(input_dim, hidden_size)
        self.b1 = np.random.rand(1, hidden_size)
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.random.rand(1, output_size)
        print('Neural net initialized with random values')

    def predict(self, X):
        """ Evaluate the network on given data batch

        np.maximum may come in handy

        Args:
        X: np.array shape (n, d)  Each row is a data point

        Output:
        pred: np.array shape (n, 1) output of network on each input point
        """
        return relu(relu(X @ self.W1 + self.b1) @ self.W2 + self.b2)

    def score(self, X, y):
        """ Compute least squares loss (1/n sum (nn(x_i) - y_i)^2)

          X: np.array shape (n, d) - Data
          y: np.array shape (n, 1) - Targets

        """
        return np.mean((self.predict(X) - y) ** 2)

# random data test
def simple_test():
    input_dim = 3
    hidden_size = 8
    N_train = 10
    X = np.random.rand(N_train, input_dim)
    y = np.random.rand(N_train, 1)
    my_net = NN(input_dim=input_dim, hidden_size=hidden_size)

    nn_out = my_net.predict(X)
    print('shape of nn_out', nn_out.shape)  # should be n x 1
    print('least squares error: ', my_net.score(X, y))

# actual data test
def housing_test():
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import fetch_california_housing
    import os, ssl
    if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
            getattr(ssl, '_create_unverified_context', None)):
        ssl._create_default_https_context = ssl._create_unverified_context

    rdata = fetch_california_housing()
    s = StandardScaler()
    Xr = rdata.data
    yr = rdata.target
    print('data size:', len(yr), 'num features:', Xr.shape[1])
    s.fit(Xr)
    X_scaled = s.transform(Xr)
    house_net = NN(input_dim=Xr.shape[1], hidden_size=8)
    weights = np.load('C:\\Users\\au544901\\Documents\\GitHub\\ML22\\week6\\good_weights.npz')
    house_net.W1 = weights['W1']
    house_net.W2 = weights['W2']
    house_net.b1 = weights['b1'].reshape(1, -1)
    house_net.b2 = weights['b2'].reshape(1, -1)
    print('hidden layer size:', house_net.W1.shape[1])
    lsq = house_net.score(X_scaled, yr.reshape(-1, 1))
    pred = house_net.predict(X_scaled)
    print('mean house price least squares error:', lsq)
    print('5 house prediction:\nestimated price , true price')
    print(np.c_[house_net.predict(X_scaled[0:10, :]), yr[0:10]])

if __name__ == '__main__':
    housing_test()