#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# import matplotlib.pyplot as plt
from nnet.data.data_utils import get_CIFAR10_data
from nnet.data.data_augmentation import random_contrast, random_flips
from nnet.data.data_augmentation import add_pad, random_crops
from nnet.data.data_augmentation import random_rotate, random_tint
from res_net import ResNet
from nnet.solver import Solver as Solver

# from nnet.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
# import matplotlib.pyplot as plt
# from nnet.utils.vis_utils import visualize_grid
import inspect
import argparse

DATASET_PATH = 'nnet/data/cifar-10-batches-py'
EXPERIMENT_PATH = '../Experiments/%d/' % 6

# residual network constants
NSIZE = 3
N_STARTING_FILTERS = 16

# solver constants
NUM_PROCESSES = 4

NUM_TRAIN = 50000
NUM_TEST = 1000

WEIGHT_DEACY = 0  # 1e-4
LEARNING_RATE = .1
MOMENTUM = .9
NUM_EPOCHS = 200
BATCH_SIZE = 128
CHECKPOINT_EVERY = 20

XH, XW = 32, 32

args = argparse.Namespace()


def parse_args():
    """
    Parse the options for running the Residual Network on CIFAR-10.
    """
    desc = ' Train a Residual Network on CIFAR-10.'
    parser = argparse.ArgumentParser(description=desc)
    add = parser.add_argument
    add('--dataset_path',
        metavar='DIRECOTRY',
        default=DATASET_PATH,
        type=str,
        help='directory where results will be saved')
    add('--experiment_path',
        metavar='DIRECOTRY',
        default=EXPERIMENT_PATH,
        type=str,
        help='directory where results will be saved')
    add('-load', '--load_checkpoint',
        metavar='DIRECOTRY',
        default='',
        type=str,
        help='load checkpoint from load_checkpoint')
    add('--n_size',
        metavar='INT',
        default=NSIZE,
        type=int,
        help='Network will have (6*n)+2 conv layers')
    add('--n_starting_filters',
        metavar='INT',
        default=N_STARTING_FILTERS,
        type=int,
        help='Network will starts with those number of filters')
    add('-np', '--n_processes',
        metavar='INT',
        default=NUM_PROCESSES,
        type=int,
        help='Number of processes for each step')
    add('--n_train',
        metavar='INT',
        default=NUM_TRAIN,
        type=int,
        help='Number of total images to select for training')
    add('--n_test',
        metavar='INT',
        default=NUM_TEST,
        type=int,
        help='Number of total images to select for validation')
    add('-wd', '--weight_decay',
        metavar='FLOAT',
        default=WEIGHT_DEACY,
        type=float,
        help='Weight decay for sgd_th')
    add('-lr', '--learning_rate',
        metavar='FLOAT',
        default=LEARNING_RATE,
        type=float,
        help='Learning rate to use with sgd_th')
    add('-mom', '--momentum',
        metavar='FLOAT',
        default=MOMENTUM,
        type=float,
        help='Nesterov momentum use with sgd_th')
    add('-nep', '--n_epochs',
        metavar='INT',
        default=NUM_EPOCHS,
        type=int,
        help='Number of epochs for training')
    add('--batch_size',
        metavar='INT',
        default=BATCH_SIZE,
        type=int,
        help='Number of images for each iteration')
    add('-cp', '--checkpoint_every',
        metavar='INT',
        default=CHECKPOINT_EVERY,
        type=int,
        help='Number of epochs between each checkpoint')
    parser.parse_args(namespace=args)


def data_augm(batch):
    p = 2
    h, w = XH, XW
    batch = random_tint(batch)
    batch = random_contrast(batch)
    batch = random_flips(batch)
    batch = random_rotate(batch, 10)
    batch = random_crops(batch, (h, w), pad=p)
    return batch


def custom_update_decay(epoch):
    if epoch in (80, 160):
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
    parse_args()

    data = get_CIFAR10_data(args.dataset_path,
                            num_training=args.n_train, num_validation=0, num_test=args.n_test)

    data = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_test'],
        'y_val': data['y_test'],
    }

    exp_path = args.experiment_path

    model = ResNet( n_size=args.n_size,
                   num_starting_filters=args.n_starting_filters,
                   hidden_dims=[],
                   reg=1e-4,
                   dtype=np.float32)

    wd = args.weight_decay
    lr = args.learning_rate
    mom = args.momentum

    optim_config = {'learning_rate': lr, 'nesterov': True,
                    'momentum': mom, 'weight_decay': wd}

    epochs = args.n_epochs
    bs = args.batch_size
    num_p = args.n_processes
    cp = args.checkpoint_every

    solver = Solver(model, data, args.load_checkpoint,
                    num_epochs=epochs, batch_size=bs,  # 20
                    update_rule='sgd_th',
                    optim_config=optim_config,
                    custom_update_ld=custom_update_decay,
                    batch_augment_func=data_augm,
                    checkpoint_every=cp,
                    num_processes=num_p)

    pretty_print(solver)
    solver.train()

    solver.export_loss(exp_path)
    solver.export_model(exp_path)
    solver.export_hostories(exp_path)
    #np.save('../Experiments/%d/model.npy' %exp, solver.model.params)
    np.save(exp_path + 'loss', solver.loss_history)
    np.save(exp_path + 'val_acc_history', solver.val_acc_history)
    np.save(exp_path + 'train_acc_history', solver.val_acc_history)

    print 'finish'

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    main()
