
import mxnet as mx
from mxnet import gluon, init, np, npx, image
from mxnet import autograd
from mxnet.gluon import nn, loss, Trainer
from read_data import *


class Accumulator:

    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""

        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for (a, b) in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


learning_rate = 0.01
devices = mx.cpu()  # #Cambiar a de GPU A CPU
pretrained_net = gluon.model_zoo.vision.vgg16(pretrained=True, ctx=devices)
        
        
net = gluon.nn.HybridSequential()  # inicializacion api sequencial
net.add(pretrained_net.features)
##net.add(gluon.nn.Dense(128, in_units=4096, activation='relu'))  # capa de salida
##net.add(gluon.nn.Dense(64, activation='relu'))  # capa de salida
##net.add(gluon.nn.Dense(32, activation='relu'))  # capa de salida
net.add(gluon.nn.Dense(4))
net[1:].initialize(init.Xavier(), ctx=devices)
net.hybridize()
loss = gluon.loss.L1Loss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate, 'wd': 0.001})




train_rmse = gluon.metric.RMSE()  # #Cambiar de gluon a mx

test_rmse = gluon.metric.RMSE()

num_epochs = 20
train_accum = Accumulator(2)
test_accum = Accumulator(2)

for epoch in range(num_epochs):
    print ('ESTOY EN LA EPOCA', epoch)
    for (i, (X_train, y_train)) in enumerate(train_data_loader):
        print ('PASE POR ACA 1')
        X_train = X_train.as_in_context(devices)
        y_train = y_train.as_in_context(devices)
        with autograd.record():
            y_pred = net(X_train)
            l = loss(y_pred, y_train)
        l.backward()
        train_rmse.update(labels=y_train, preds=y_pred)
        trainer.step(batch_size)
        rmse_val = train_rmse.get()
        if (i % len(train_data_loader) // 10) == 0:
            print ('epoch : {}, train loss : {} , train RMSE : {}'.format(epoch, l.asnumpy().mean(), train_rmse.get()))

    for (X_test, y_test) in test_data_loader:
        print ('PASE POR ACA 2')
        X_test = X_test.as_in_context(devices)
        y_test = y_test.as_in_context(devices)
        y_pred = net(X_test)
        test_rmse.update(labels=y_test, preds=y_pred)
        l = loss(net(X_test), y_test)
        rmse_val = test_rmse.get()
        print ('epoch : {}, test loss : {}, train RMSE : {} '.format(epoch, l.asnumpy().mean(), test_rmse.get()))
                

print ('-------------------------------------')
