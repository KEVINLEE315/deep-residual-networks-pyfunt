#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
from nnet.data.data_utils import get_CIFAR10_data
from nnet.data.data_augmentation import random_tint, random_contrast, random_flips, add_pad, random_crops
from res_net import ResNet
from nnet.solver import Solver as Solver

# from nnet.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
# import matplotlib.pyplot as plt
# from nnet.utils.vis_utils import visualize_grid
import inspect

# residual network constants
NSIZE = 3

'''
        .___          /\ __      __                       .__        __  .__    .__
      __| _/____   ___)//  |_  _/  |_  ____  __ __   ____ |  |__   _/  |_|  |__ |__| ______
     / __ |/  _ \ /    \   __\ \   __\/  _ \|  |  \_/ ___\|  |  \  \   __\  |  \|  |/  ___/
    / /_/ (  <_> )   |  \  |    |  | (  <_> )  |  /\  \___|   Y  \  |  | |   Y  \  |\___ \
    \____ |\____/|___|  /__|    |__|  \____/|____/  \___  >___|  /  |__| |___|  /__/____  >
         \/           \/                                \/     \/             \/        \/
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
'''
N_STARTING_FILTERS = 16

# solver constants
NUM_PROCESSES = 4

WEIGHT_DEACY = 1e-4
LEARNING_RATE = .1
MOMENTUM = .9
NUM_EPOCHS = 160
BATCH_SIZE = 128
CHECK_POINT_EVERY = 20
VERBOSE = True
PRINT_EVERY = False

XH, XW = 32, 32

assert(VERBOSE and PRINT_EVERY) or not(VERBOSE and PRINT_EVERY)

DATA = get_CIFAR10_data(
    num_training=50000, num_validation=0, num_test=1000)

num_train = 50000
DATA = {
    'X_train': DATA['X_train'][:num_train],
    'y_train': DATA['y_train'][:num_train],
    'X_val': DATA['X_test'],
    'y_val': DATA['y_test'],
}

def data_augm(batch):
    p = 2
    h, w = XH, XW
    batch = random_tint(batch)
    batch = random_contrast(batch)
    batch = random_flips(batch)
    batch = add_pad(batch, p)
    batch = random_crops(batch, (h, w))
    return batch


def custom_update_decay(epoch):
    if epoch in (80, 120):
        return 0.1
    return 1


def pretty_print(solver):
    print 'Model: \n%s' % solver.model

    print 'Solver: \n%s' % solver

    print 'Data Augmentation Function: \n'
    print ''.join(['\t' + i for i in inspect.getsourcelines(data_augm)[0]])
    print 'Custom Weight Decay Update Rule: \n'
    print ''.join(['\t' + i for i in inspect.getsourcelines(custom_update_decay)[0]])


def main():
    model = ResNet(
        n_size=NSIZE, num_starting_filters=N_STARTING_FILTERS, hidden_dims=[64,64], dtype=np.float32)

    wd = WEIGHT_DEACY
    lr = LEARNING_RATE
    mom = MOMENTUM

    optim_config = {'learning_rate': lr, 'nesterov': True,
                    'momentum': mom, 'weight_decay': wd}

    data = DATA
    epochs = NUM_EPOCHS
    bs = BATCH_SIZE
    nump = NUM_PROCESSES
    cp = CHECK_POINT_EVERY
    v = VERBOSE
    pe = PRINT_EVERY

    solver = Solver(model, data,
                    num_epochs=epochs, batch_size=bs,  # 20
                    update_rule='sgd_th',
                    optim_config=optim_config,
                    verbose=v, print_every=pe,
                    custom_update_ld=custom_update_decay,
                    batch_augment_func=data_augm,
                    check_point_every=cp,
                    num_processes=nump)

    pretty_print(solver)
    solver.train()
    solver.export_model()
    solver.export_loss()

    print 'finish'

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
