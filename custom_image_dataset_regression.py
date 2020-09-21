# -*- coding: utf-8 -*-
# !pip install mxnet gluoncv

import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data.vision import transforms
import multiprocessing
import numpy as np
from mxnet import nd, init, autograd
from mxnet.gluon import nn
import time

class MxnetDataCreator(object):
  def __init__(self, X_train, X_test, Y_train, Y_test):
    self.X_tr = X_train
    self.X_te = X_test
    self.Y_tr = Y_train
    self.Y_te = Y_test

  def convert_to_mx_nd(self):
    self.X_tr = mx.nd.array(self.X_tr)
    self.X_te = mx.nd.array(self.X_te)
    
  def create_mx_dataset(self):
    self.convert_to_mx_nd()
    train = mx.gluon.data.ArrayDataset(self.X_tr, self.Y_tr)
    test = mx.gluon.data.ArrayDataset(self.X_te, self.Y_te)
    return train, test

class MxnetDataLoader(object):
  def __init__(self):
    pass

  def create_loader(self, train_data, test_data, batch_size=4, transformer=None):
    if transformer:

      train_data, test_data = self.transform(train_data, test_data, transformer)
      workers = multiprocessing.cpu_count()
      train_data = gluon.data.DataLoader(
      train_data, batch_size=batch_size, shuffle=True, num_workers=workers)

      test_data = gluon.data.DataLoader(
      test_data, batch_size=batch_size, shuffle=True, num_workers=workers)

    else:

      transformer = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize(0.13, 0.31)])

      train_data, test_data = self.transform(train_data, test_data, transformer)

      workers = multiprocessing.cpu_count()
      train_data = gluon.data.DataLoader(
      train_data, batch_size=batch_size, shuffle=True, num_workers=workers)

      test_data = gluon.data.DataLoader(
      test_data, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    return train_data, test_data
  
  @staticmethod
  def transform(train_data, test_data, transformer):
      train_data = train_data.transform_first(transformer)
      test_data = test_data.transform_first(transformer)
      return train_data, test_data

# load a non mxnet dataset.

X_train = np.random.rand(20000, 64, 64, 3)
X_test = np.random.rand(3000, 64, 64, 3)
Y_train = np.random.randint(-90, 90, size=(20000, 3))
Y_test = np.random.randint(-90, 90, size=(3000, 3))
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(Y_test[0:10])

# use the utilities to easily manage the data.
prep = MxnetDataCreator(X_train, X_test, Y_train, Y_test)
dl = MxnetDataLoader()
train_data, test_data = prep.create_mx_dataset()
X, y = train_data[0]
('X shape: ', X.shape, 'X dtype', X.dtype, 'y:', y)
train_data, test_data = dl.create_loader(train_data, test_data, batch_size=256,
                                         transformer=None)
for data, label in train_data:
    print(data.shape, label.shape)
    break

# create a cnn.
net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(channels=16, kernel_size=3, activation='relu'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Flatten(),
        nn.Dense(120, activation="relu"),
        nn.Dense(84, activation="relu"),
        nn.Dense(3))
net.initialize(init=init.Xavier())

def square_loss(yhat, y): 
    return mx.nd.mean((yhat - y) ** 2)

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
batch_size=256

for epoch in range(10):
    train_loss, valid_loss = 0., 0.
    tic = time.time()
    for data, label in train_data:
        # forward + backward
        with autograd.record():
            output = net(data)
            loss = square_loss(output.astype(int), label.astype(int))
        loss.backward()
        # update parameters
        trainer.step(batch_size)
        # calculate training metrics
        train_loss += loss.mean().asscalar()
    # calculate validation accuracy
    for data, label in test_data:
        output = net(data)
        loss += square_loss(output.astype(int), label.astype(int))
        valid_loss += loss.mean().asscalar()
    print("Epoch %d: train loss %.3f, test loss %.3f in %.1f sec" % (
            epoch+1, train_loss/len(train_data), valid_loss/len(test_data),
             time.time()-tic))

net.save_parameters('net.params')
