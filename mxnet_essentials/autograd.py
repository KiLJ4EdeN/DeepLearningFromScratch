# -*- coding: utf-8 -*-
# !pip install mxnet gluoncv

import mxnet as mx
from mxnet import nd, autograd
mx.random.seed(1)

# target is differentiating f = 2 * (x ** 2)
x = nd.array([[1, 2], [3, 4]])

# we want to store gradients
x.attach_grad()

# start recording.
with autograd.record():
    y = x * 2
    z = y * x

z.backward()

# 2x ** 2
# 4x
print(x.grad)

with autograd.record():
    y = x * 2
    z = y * x

head_gradient = nd.array([[10, 1.], [.1, .01]])
z.backward(head_gradient)
print(x.grad)

a = nd.random_normal(shape=3)
a.attach_grad()

with autograd.record():
    b = a * 2
    while (nd.norm(b) < 1000).asscalar():
        b = b * 2

    if (mx.nd.sum(b) > 0).asscalar():
        c = b
    else:
        c = 100 * b

head_gradient = nd.array([0.01, 1.0, .1])
c.backward(head_gradient)

print(a.grad)
