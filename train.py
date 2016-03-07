#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from res_net import ResNet
from nnet.data.data_utils import *
from nnet.data.data_augmentation import random_flips, add_pad, random_crops
# from nnet.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from nnet.solver import Solver as Solver
import matplotlib.pyplot as plt
from nnet.utils.vis_utils import visualize_grid

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def rel_error(x, y): return np.max(
    np.abs(x - y) / (np.maximum(1e-6, np.abs(x) + np.abs(y))))


data = get_CIFAR10_data(
    num_training=50000, num_validation=0, num_test=1000)

num_train = 50000
data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_test'],
    'y_val': data['y_test'],
}


def data_augm(batch):
    batch = random_flips(batch)
    batch = add_pad(batch, 2)
    batch = random_crops(batch, (32, 32))
    return batch


def custom_update_decay(epoch):
    # if epoch == 1:
        # return 10
    if epoch in (80, 120):
       return 0.1
    return 1


if __name__ == '__main__':
    #model = ResNet(n_size=1, nfilters=16, naffines=0, dtype=np.float32)
    model = ResNet(n_size=1, num_starting_filters=16)

    wd = 1e-4
    optim_config = {'learning_rate': .1, 'nesterov': True,
                    'momentum': .9, 'weight_decay': wd}


    name = 'resnet3N'
    solver = Solver(model, data,
                    num_epochs=160, batch_size=64,  # 20
                    update_rule='sgd_th',
                    optim_config=optim_config,
                    verbose=True, print_every=100,
                    custom_update_ld=custom_update_decay,
                    batch_augment_func=data_augm,
                      check_point_every=20,
                    num_processes=1)


    print 'Model: ' + name + ' ' + str(model)

    print 'Solver: ' + str(solver)

    solver.train()
    print 'finish'

    import pdb
    pdb.set_trace()

    plt.subplot(2, 1, 1)
    plt.plot(solver.loss_history)
    plt.ylabel('loss')
    plt.xlabel('Iteration')

    plt.subplot(2, 1, 2)
    plt.plot(solver.train_acc_history)
    plt.plot(solver.val_acc_history)
    plt.ylabel('accuracy')
    plt.xlabel('Check')
    plt.legend(['train', 'val'], loc='upper left')

    plt.show()

