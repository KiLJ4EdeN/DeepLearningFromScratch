# -*- coding: utf-8 -*-
# !pip install mxnet gluoncv

import mxnet as mx

mx.random.seed(1)

x = mx.nd.empty(shape=(3, 4))
x

x = mx.nd.zeros((3, 4))
x

# random normal dist with zero mean and one std.
y = mx.nd.random_normal(0, 1, shape=(3, 4))
y

print(y.shape)
print(y.size)

x + y

mx.nd.exp(y)

mx.nd.dot(x.T, y)

# the memory place is allocated again for y.

print('id(y):', id(y))
y = y + x
print('id(y):', id(y))

# this way memory allocation is not done, when using mx.nd arrays.

print('id(y):', id(y))
y[:] = x + y
print('id(y):', id(y))

mx.nd.elemwise_add(x, y, out=y)

print('id(x):', id(x))
x += y
x
print('id(x):', id(x))

print(x[1:3])
x[1,2] = 9.0
x

x[1:2,1:3]

x[1:2,1:3] = 5.0
x

x = mx.nd.ones(shape=(3,3))
print('x = ', x)
y = mx.nd.arange(3)
print('y = ', y)
print('x + y = ', x + y)

y = y.reshape((3,1))
print('y = ', y)
print('x + y = ', x+y)

a = x.asnumpy()
type(a)

y = mx.nd.array(a) 
y

z = mx.nd.ones(shape=(3,3), ctx=mx.cpu(0))
z

# move to another device.

x_gpu = x.copyto(mx.gpu(0))
print(x_gpu)

print(z.context)

# with blind copying more memory is occupied each time.
# can be prevented with as_in_context.

print('id(z):', id(z))
z = z.copyto(mx.cpu(0))
print('id(z):', id(z))
z = z.as_in_context(mx.cpu(0))
print('id(z):', id(z))
print(z)
