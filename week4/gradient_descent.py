import numpy as np
import matplotlib.pyplot as plt

def f(a, x):
    return 0.5 * (x[0]**2 + a * x[1]**2)

def fp(a, x):
    return np.array([x[0], a * x[1]])

def visualize(a, path, ax=None):
    """
    Make contour plot of f_a and plot the path on top of it
    """
    y_range = 10
    x = np.arange(-257, 257, 0.1)
    y = np.arange(-y_range, y_range, 0.1)
    xx, yy = np.meshgrid(x, y)
    z = 0.5 * (xx**2 + a * yy**2)
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 13))
    h = ax.contourf(xx, yy, z, cmap=plt.get_cmap('jet'))
    ax.plot([x[0] for x in path], [x[1] for x in path], 'w.--', markersize=4)
    ax.plot([0], [0], 'rs', markersize=8) # optimal solution
    ax.set_xlim([-257, 257])
    ax.set_ylim([-y_range, y_range])

def gd(a, step_size=0.1, steps=40):
    """ Run Gradient descent
        params:
        a - the parameter that define the function f
        step_size - constant stepsize to use for gradient descent
        steps - number of steps to run

        Returns: out, list with the sequence of points considered during the descent.
    """
    x = np.array([256.0, 1.0])  # starting point
    out = [x]
    for _ in range(steps):
        x = x - step_size * fp(a, x)
        out.append(x)
    return out


fig, axes = plt.subplots(2, 3)
ateam = [[1, 4, 16], [20, 30, 40]]
for i in range(2):
    for j in range(3):
        ax = axes[i][j]
        a = ateam[i][j]
        path = gd(a, step_size=1/a, steps=10*a)  # use good step size here instead of standard value
        visualize(a, path, ax)
        ax.set_title('Gradient Descent a={0}'.format(a))