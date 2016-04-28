#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import numpy as np
from res_net import ResNet
# from nnet.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from pyfunt.solver import Solver as Solver
import matplotlib.pyplot as plt
from pyfunt.utils.vis_utils import visualize_grid

plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

exp_path = '../Experiments/%s/' % str(uuid.uuid4())[-10:]


def show_weights(path):
    plt.grid(True, color='w', linestyle='-', linewidth=0)
    w = np.load('%s/model.npy' % path).item()['W1']

    grid = visualize_grid(w.transpose(0, 2, 3, 1))
    plt.tick_params(axis='both', which='both', bottom='off', top='off',
                    labelbottom='off', right='off', left='off', labelleft='off')
    plt.show(grid.astype('uint8'))


if __name__ == '__main__':
    model = ResNet(n_size=1, num_starting_filters=16)

    wd = 1e-4
    optim_config = {'learning_rate': .1, 'nesterov': True,
                    'momentum': .9, 'weight_decay': wd}

    name = 'resnet3N'
    solver = Solver(model, load_dir='checkpoints')

    solver.export_model(exp_path)
    solver.export_histories(exp_path)
    show_weights(exp_path)

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
