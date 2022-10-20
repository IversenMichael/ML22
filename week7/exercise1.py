import numpy as np
import matplotlib.pyplot as plt
from PIL import Image #(package name is pillow - i.e. pip3 install pillow )
import os

def main():
    filename = 'C:\\Users\\au544901\\Documents\\GitHub\\ML22\\week7\\tiger.bmp'
    with Image.open(filename) as img:
        tiger = np.array(img)
        print('image shape', tiger.shape)
        plt.imshow(tiger, cmap='gray')
        plt.show()
        conv_filter = np.array([
            [-1., -1., -1.],
            [-1., 8, -1.],
            [-1., -1., -1.]
        ])
        convoluted_tiger = conv2d(tiger, conv_filter)
        pooled_tiger = max_pool2d(convoluted_tiger)
        plot_min = convoluted_tiger.mean()-convoluted_tiger.std()
        plot_max = convoluted_tiger.mean()+convoluted_tiger.std()
        fig, axes = plt.subplots(1, 2, figsize=(20, 16))
        axes[0].imshow(convoluted_tiger, cmap='gray', vmin=plot_min, vmax=plot_max)
        axes[1].imshow(pooled_tiger, cmap='gray', vmin=plot_min, vmax=plot_max)
        plt.show()

        with_torch(tiger, convoluted_tiger, pooled_tiger)

def conv2d(img, w):
    """ Return the result of applying the convolution defined by w to img -
    for simplicity assume that w is square"""
    w_dim = w.shape[0]
    pad = w_dim - 2
    padded_img = np.pad(img, [pad, pad], 'constant', constant_values=0)
    out = np.zeros(img.shape)
    ### YOUR CODE HERE
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.sum(padded_img[i:(i + 3), j:(j + 3)] * w)
    ### END CODE
    return out

def max_pool2d(img):
    """ Return the result of applying the 2 x 2 max pooling operator to mig (halve the width and height of image)"""
    out = np.zeros((int(img.shape[0]/2), int(img.shape[1]/2)))
    ### YOUR CODE HERE
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.max(img[(2 * i):(2 * i + 2), (2 * j):(2 * j + 2)])
    ### END CODE
    return out

def with_torch(tiger, convoluted_tiger, pooled_tiger):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch_tiger = torch.from_numpy(tiger).view(1, 1, tiger.shape[0], tiger.shape[1]).double()
    print('original',tiger.min(),tiger.max())
    print('image shape', torch_tiger.shape)
    tv = torch.tensor([[-1., -1., -1.], [-1., 8, -1.], [-1., -1., -1.]])
    tv = tv.view(1, 1, 3, 3).double()
    torch_convoluted_tiger = F.conv2d(torch_tiger, tv, torch.tensor([0.], dtype=torch.double), 1, 1, 1, 1)
    numpy_convoluted_tiger = torch_convoluted_tiger.numpy().squeeze()
    print('convoluted_tiger shape', numpy_convoluted_tiger.shape)
    print('conv diff norm', np.linalg.norm(numpy_convoluted_tiger - convoluted_tiger))
    torch_pooled_tiger = F.max_pool2d(torch_convoluted_tiger, kernel_size=(2, 2))
    numpy_pooled_tiger = torch_pooled_tiger.numpy().squeeze()
    print('pool diff norm', np.linalg.norm(numpy_pooled_tiger - pooled_tiger))
    fig, axes = plt.subplots(1, 2, figsize=(20, 16))
    plot_data_min = numpy_convoluted_tiger.mean()-numpy_convoluted_tiger.std()
    plot_data_max = numpy_convoluted_tiger.mean()+numpy_convoluted_tiger.std()
    axes[0].imshow(numpy_convoluted_tiger, cmap='gray', vmin=plot_data_min, vmax=plot_data_max)
    axes[1].imshow(numpy_pooled_tiger, cmap='gray', vmin=plot_data_min, vmax=plot_data_max)
    plt.show()

if __name__ == '__main__':
    main()