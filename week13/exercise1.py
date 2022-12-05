# ## PCA on digits
# In this exercise we will experiment with PCA on digits.

# 1. Run PCA on the training data train_dat http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html (use n_components = 784) and use ``plot_images`` to plot the 20 directions with largest variance. Use this PCA model for all subsequent computations. The directions can be found with the attribute ``components_``.
# 2. Take the first 20 data points from the training data and project them onto the first $k$ components for $k \in \{1, 2,4,8,16, 32, 64\}$. Then plot them as images. What do you see?
#     **Hint:** To project them compute the length of the projection of each point onto the first $k$ directions and then compute for each image compute the linear combination of the first $k$ directions given by these directions (XZZ^T as in lecture).
# 3. Map all the training data train_dat to 2D (the length of the projection on the first two directions) and make a scatter plot where you color with the label and see if there is some structure (use scatter with ``cmap = plt.cm.Paired``, like ``ax.scatter(x,y, c=lab, cmap=plt.cm.Paired)``)
# 4. Map all training data train_dat to $32$ dimensions and train an SGD Classifier. Observe the test accuracy of the classifier. How should you select target dimension in real life?
#    Use the following classifer (svm loss): ``clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200, fit_intercept=True)``

# Discuss the results. Are they what you expect? Why? Why not.


# %matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
import torchvision.datasets as datasets

# Load full dataset
mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)


def plot_images(dat, k=20, size=28):
    """ Plot the first k vectors as 28 x 28 images """
    x2 = dat[0:k, :].reshape(-1, size, size)
    x2 = x2.transpose(1, 0, 2)
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.imshow(x2.reshape(size, -1), cmap='bone')
    ax.set_yticks([])
    ax.set_xticks([])
    plt.show()


# data, labels =
data = mnist_trainset.data.numpy().reshape((60000, 28 * 28))
labels = mnist_trainset.targets.numpy()

# reduce size for speed
rp = np.random.permutation(len(labels))
train_dat = data[rp[:5000], :]
test_dat = data[rp[5000:6000], :]
train_lab = labels[rp[:5000]]
test_lab = labels[rp[5000:6000]]

components = None
## TASK 1
### YOUR CODE HERE
### END CODE
print('First 20 directions (eigenvectors X^T X)')
plot_images(components, 20)

## TASK 2
# take the first 20 data point and project them onto the first k components and plot for k in [1, 2,4,8,16, 32, 64]
img = train_dat[0:20, :]
print('Original images:')
plot_images(img, 20)
for k in [1, 2, 4, 8, 16, 32, 64]:
    ...
### YOUR CODE HERE
### END CODE

# map the data to 2D and plot the results colored by label
proj = None
## TASK 3
### YOUR CODE HERE
### END CODE
fig, ax = plt.subplots(figsize=(20, 12))
ax.scatter(proj[:, 0], proj[:, 1], c=train_lab, cmap=plt.cm.Paired)
plt.show()

## TASK 4
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200, fit_intercept=True)
### YOUR CODE HERE
### END CODE

clf_original = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200, fit_intercept=True)
clf_original.fit(train_dat, train_lab)

proj_test_dat = test_dat @ components[0:32].T
acc = (clf.predict(proj_test_dat) == test_lab).mean()
print('Test Accuracy of 32 Dim PCA:', 100 * acc)
print('Test Accuracy original:', 100 * ((clf_original.predict(test_dat) == test_lab).mean()))