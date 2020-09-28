# naive bayes
# key assumption: all attributes are independent of each other.

"""
The sum of the probabilty of each feature associated with the wanted label.
P(x|y) = sum(P(x_i|y))

Using Bayes Theorem We could say that:
=> P(y|x) = sum(P(x_i|y)) * p(y) / p(x)
This is the formula for the naive bayes classfier.
Here we dont know 'p(x)'. But we can have a workaround with

sum(P(y|x)) = 1 (sum over y)
the probabilty over all the labels must be one so p(x) is not needed.
"""

# mnist classification with naivebayes.

import numpy as np
import mxnet as mx
from mxnet import nd
import matplotlib.pyplot as plt

# we go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the count statistics for p(y) and p(x_i|y)
# We initialize all numbers with a count of 1 to ensure that we don't get a
# division by zero.  Statisticians call this Laplace smoothing.

# ycount is probabilities for each class. p(y)
# its calculated by computing the prevelance of each class.
# ex: class 4 (digit 4) occurs 5800 times. the probabilty for the class is:
# => 5800/60000 where 60000 is the total number of images.
ycount = nd.ones(shape=(10))

# xcount is the probabilty of each pixels for seperate classes.
xcount = nd.ones(shape=(784, 10))


# Aggregate count statistics of how frequently a pixel is on (or off) for
# zeros and ones.
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    # one time occurance of class is added.
    ycount[y] += 1
    # the class column is updated for this class, using all of the pixels.
    xcount[:, y] += x

# normalize the probabilities p(x_i|y) (divide per pixel counts by total
# count)
for i in range(10):
    # total_pixels / total_sample_count
    xcount[:, i] = xcount[:, i] / ycount[i]

# likewise, compute the probability p(y)
py = ycount / nd.sum(ycount)


# plot the calculated probabilities.
fig, figarr = plt.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print(py)

"""
That is, instead of p(x|y) = prod_{i} p(x_i|y) we compute:
                     log p(x|y) = sum_i log p(x_i|y).
                     
$$l_y := \sum_i \log p(x_i|y) = \sum_i x_i \log p(x_i = 1|y) + (1-x_i) \log \left(1-p(x_i=1|y)\right)$$

To avoid recomputing logarithms all the time, we precompute them for all pixels.
"""

logxcount = nd.log(xcount)
logxcountneg = nd.log(1-xcount)
logpy = nd.log(py)

fig, figarr = plt.subplots(2, 10, figsize=(15, 3))

# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)
    
    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] += nd.dot(logxcount[:, i], x) + nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= nd.max(logpx)
    # and compute the softmax using logpx
    px = nd.exp(logpx).asnumpy()
    px /= np.sum(px)

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break

plt.show()
