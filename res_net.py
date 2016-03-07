#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from nnet.layers.layer import skip_forward, skip_backward
from nnet.layers.layers import log_softmax_loss
from nnet.layers.layer_utils import conv_batchnorm_relu_forward
from nnet.layers.layer_utils import conv_batchnorm_relu_backward
from nnet.layers.layer_utils import affine_batchnorm_relu_forward
from nnet.layers.layer_utils import affine_batchnorm_relu_backward
from nnet.layers.layer_utils import avg_pool_forward
from nnet.layers.layer_utils import avg_pool_backward
from nnet.layers.layer_utils import affine_forward, affine_backward
from nnet.layers.init import init_conv_w_kaiming, init_bn_w_gcr
from nnet.layers.init import init_affine_wb


class ResNet(object):

    '''
    Implementation of ["Deep Residual Learning for Image Recognition",Kaiming \
    He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - http://arxiv.org/abs/1512.03385

    Inspired by https://github.com/gcr/torch-residual-networks

    This network should model a similiar behaviour of gcr's implementation.
    Check https://github.com/gcr/torch-residual-networks for more infos about \
    the structure.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.

    The network has, like in the reference paper, (6*n)+2 layers,
    composed as below:
                                                (image_dim: 3, 32, 32; F=16)
                                                (input_dim: N, *image_dim)
         INPUT
            |
            v
       +-------------------+
       |conv[F, *image_dim]|                    (out_shape: N, 16, 32, 32)
       +-------------------+
            |
            v
       +-------------------------+
       |n * res_block[F, F, 3, 3]|              (out_shape: N, 16, 32, 32)
       +-------------------------+
            |
            v
       +-------------------------+
       |res_block[2*F, F, 3, 3]  |              (out_shape: N, 32, 16, 16)
       +-------------------------+
            |
            v
       +---------------------------------+
       |(n-1) * res_block[2*F, 2*F, 3, 3]|      (out_shape: N, 32, 16, 16)
       +---------------------------------+
            |
            v
       +-------------------------+
       |res_block[4*F, 2*F, 3, 3]|              (out_shape: N, 64, 8, 8)
       +-------------------------+
            |
            v
       +---------------------------------+
       |(n-1) * res_block[4*F, 4*F, 3, 3]|      (out_shape: N, 64, 8, 8)
       +---------------------------------+
            |
            v
       +-------------+
       |pool[1, 8, 8]|                          (out_shape: N, 64, 1, 1)
       +-------------+
            |
            v
       +-------+
       |softmax|                                (out_shape: N, num_classes)
       +-------+
            |
            v
         OUTPUT

    Every convolution layer has a pad=1 and stride=1, except for the  dimension
    enhancning layers which has a stride of 2 to mantain the computational
    complexity.
    Optionally, there is the possibility of setting m affine layers immediatley
    before the softmax layer by setting the hidden_dims parameter, which should
    be a list of integers representing the numbe of neurons for each affine
    layer.

    Each residual block is composed as below:

                  Input
                     |
             ,-------+-----.
       Downsampling      3x3 convolution+dimensionality reduction
            |               |
            v               v
       Zero-padding      3x3 convolution
            |               |
            `-----( Add )---'
                     |
                  Output

    After every layer, a batch normalization with momentum .1 is applied.

    Weight initialization (check also layers/init.py and layers/README.md):
    - Inizialize the weights and biases for the affine layers in the same
     way of torch's default mode by calling _init_affine_wb that returns a
     tuple (w, b).
    - Inizialize the weights for the conv layers in the same
     way of torch's default mode by calling init_conv_w.
    - Inizialize the weights for the conv layers in the same
     way of kaiming's mode by calling init_conv_w_kaiming
     (http://arxiv.org/abs/1502.01852 and
      http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-\
      initialization)
    - Initialize batch normalization layer's weights like torch's default by
    calling init_bn_w
    - Initialize batch normalization layer's weights like cgr's first resblock\
    's bn (https://github.com/gcr/torch-residual-networks/blob/master/residual\
           -layers.lua#L57-L59) by calling init_bn_w_gcr.

    '''

    def __init__(self, input_dim=(3, 32, 32), num_starting_filters=16, n_size=1,
                 hidden_dims=[], num_classes=10, reg=0.0, dtype=np.float32):
        '''
        num_filters=[16, 16, 32, 32, 64, 64],
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data.
        - num_starting_filters: Number of filters for the first convolution
        layer.
        - n_size: nSize for the residual network like in the reference paper
        - hidden_dims: Optional list number of units to use in the
        fully-connected hidden layers between the fianl pool and the sofmatx
        layer.
        - num_classes: Number of scores to produce from the final affine layer.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        '''
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        self.bn_params = {}
        self.n_size = n_size
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.filter_size = 3

        self.num_filters = self._init_filters(num_starting_filters)
        self.L = len(self.num_filters)  # Number of convs
        self.M = len(hidden_dims)  # Number of affines

        self._init_conv_weights()

        self.pool_param2 = {'stride': 8, 'pool_height': 8, 'pool_width': 8}

        self.h_dims = (
            [hidden_dims[0]] if len(hidden_dims) > 0 else []) + hidden_dims

        self._init_affine_weights()

        self._init_scoring_layer(num_classes)

        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)

    def __str__(self):
        return """
        Residual Network:
            NSize: %d;
            Numbers of filters for each layer:\n %d;
            Optional linear layers dimensions: %s;
        """ % (self.n_size,
               self.num_filters[0],
               str(self.hidden_dims))

    def _init_filters(self, nf):
        '''
        Initialize conv filters like in
        https://github.com/gcr/torch-residual-networks
        Called by self.__init__
        '''

        num_filters = [nf]  # first conv
        for i in range(self.n_size):  # n res blocks
            num_filters += [nf] * 2
        nf *= 2
        num_filters += [nf] * 2  # res block increase ch
        for i in range(self.n_size-1):  # n-1 res blocks
            num_filters += [nf] * 2

        nf *= 2
        num_filters += [nf] * 2  # res block increase ch
        for i in range(self.n_size-1):  # n-1 res blocks
            num_filters += [nf] * 2
        return num_filters

    def _init_conv_weights(self):
        '''
        Initialize conv weights.
        Called by self.__init__
        '''
        # Size of the input
        Cinput, Hinput, Winput = self.input_dim
        filter_size = self.filter_size

        # Initialize the weight for the conv layers
        F = [Cinput] + self.num_filters
        for i in xrange(self.L):
            idx = i + 1
            shape = F[i + 1], F[i], filter_size, filter_size
            out_ch = shape[0]
            W = init_conv_w_kaiming(shape)
            b = np.zeros(out_ch)
            self.params['W%d' % idx] = W
            self.params['b%d' % idx] = b
            bn_param = {'mode': 'train',
                        'running_mean': np.zeros(out_ch),
                        'running_var': np.ones(out_ch)}
            gamma = init_bn_w_gcr(out_ch)
            beta = np.zeros(out_ch)
            self.bn_params['bn_param%d' % idx] = bn_param
            self.params['gamma%d' % idx] = gamma
            self.params['beta%d' % idx] = beta

        # Initialize conv/pools parameters
        self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
        self.conv_param2 = {'stride': 2, 'pad': 0}
        self.pool_param1 = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    def _init_affine_weights(self):
        '''
        Initialize affine weights.
        Called by self.__init__
        '''
        dims = self.h_dims
        for i in xrange(self.M):
            idx = self.L + i + 2
            shape = dims[i], dims[i + 1]
            out_ch = shape[1]
            W, b = init_affine_wb(shape)
            self.params['W%d' % idx] = W
            self.params['b%d' % idx] = b
            bn_param = {'mode': 'train',
                        'running_mean': np.zeros(out_ch),
                        'running_var': np.ones(out_ch)}
            gamma = np.ones(out_ch)
            beta = np.zeros(out_ch)
            self.bn_params['bn_param%d' % idx] = bn_param
            self.params['gamma%d' % idx] = gamma
            self.params['beta%d' % idx] = beta

    def _init_scoring_layer(self, num_classes):
        '''
        Initialize scoring layer weights.
        Called by self.__init__
        '''
        # Scoring layer
        in_ch = self.h_dims[-1] if \
            len(self.h_dims) > 0 else self.num_filters[-1]
        shape = in_ch, num_classes
        W, b = init_affine_wb(shape)
        i = self.L + self.M + 2
        self.params['W%d' % i] = W
        self.params['b%d' % i] = b

    def loss_helper(self, args):
        '''
        Helper method used to call loss() within a pool of processes using \
        pool.map_async.
        '''
        return self.loss(*args)

    def loss(self, X, y=None):
        '''
        TODO: split in _functions
        Evaluate loss and gradient for the three-layer convolutional network.

        '''
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        params = self.params
        conv_param1 = self.conv_param
        conv_param2 = self.conv_param2
        pool_param2 = self.pool_param2
        for key, bn_param in self.bn_params.iteritems():
            bn_param[mode] = mode
        scores = None

        blocks = {}
        blocks['h0'] = X
        # Forward into the conv blocks
        for i in xrange(self.L):
            idx = i + 1
            w = params['W%d' % idx]
            out_ch, in_ch = w.shape[:2]
            b = params['b%d' % idx]
            h = blocks['h%d' % (idx - 1)]
            if i > 0 and i % 2 == 1:
                # store skip
                skip, skip_cache = skip_forward(h, out_ch)
                blocks['cache_skip%d' % idx] = skip_cache
            beta = params['beta%d' % idx]
            gamma = params['gamma%d' % idx]
            bn_param = self.bn_params['bn_param%d' % idx]
            if i == 0 or in_ch == out_ch:
                conv_param = conv_param1
            else:
                conv_param = conv_param2
                h = np.pad(
                    h, ((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant')

            h, cache_h = conv_batchnorm_relu_forward(
                h, w, b, conv_param, gamma, beta, bn_param)

            if i > 0 and i % 2 == 0:
                # add skip
                h += skip
            blocks['h%d' % idx] = h
            blocks['cache_h%d' % idx] = cache_h

        # Pool
        idx = self.L + 1
        h = blocks['h%d' % (idx - 1)]
        h, cache_h = avg_pool_forward(h, pool_param2)
        blocks['h%d' % idx] = h
        blocks['cache_h%d' % idx] = cache_h

        # Forward into the linear blocks
        for i in xrange(self.M):
            idx = self.L + i + 2
            h = blocks['h%d' % (idx - 1)]

            w = params['W%d' % idx]
            b = params['b%d' % idx]
            beta = params['beta%d' % idx]
            gamma = params['gamma%d' % idx]
            bn_param = self.bn_params['bn_param%d' % idx]
            h, cache_h = affine_batchnorm_relu_forward(h, w, b, gamma,
                                                       beta, bn_param)

        # Fnally Forward into the score
        idx = self.L + self.M + 2
        w = params['W%d' % idx]
        b = params['b%d' % idx]
        h = blocks['h%d' % (idx - 1)]
        h, cache_h = affine_forward(h, w, b)
        blocks['h%d' % idx] = h
        blocks['cache_h%d' % idx] = cache_h

        scores = blocks['h%d' % idx]

        if y is None:
            return scores

        loss, grads = 0, {}

        # Computing of the loss
        data_loss, dscores = log_softmax_loss(scores, y)
        reg_loss = 0
        for w in [params[f] for f in params.keys() if f[0] == 'W']:
            reg_loss += 0.5 * self.reg * np.sum(w * w)

        loss = data_loss + reg_loss

        # Backward pass
        # print 'Backward pass'
        # Backprop into the scoring layer
        idx = self.L + self.M + 2
        dh = dscores
        h_cache = blocks['cache_h%d' % idx]
        dh, dw, db = affine_backward(dh, h_cache)
        blocks['dh%d' % (idx - 1)] = dh
        blocks['dW%d' % idx] = dw
        blocks['db%d' % idx] = db

        # Backprop into the linear blocks
        for i in range(self.M)[::-1]:
            idx = self.L + i + 2
            dh = blocks['dh%d' % idx]
            h_cache = blocks['cache_h%d' % idx]
            dh, dw, db, dgamma, dbeta = affine_batchnorm_relu_backward(
                dh, h_cache)
            blocks['dbeta%d' % idx] = dbeta
            blocks['dgamma%d' % idx] = dgamma

        # back pool
        idx = self.L + 1
        dh = blocks['dh%d' % idx]
        h_cache = blocks['cache_h%d' % idx]
        dh = avg_pool_backward(dh, h_cache)
        blocks['dh%d' % (idx - 1)] = dh

        # Backprop into the conv blocks
        for i in range(self.L)[::-1]:
            idx = i + 1
            dh = blocks['dh%d' % idx]
            h_cache = blocks['cache_h%d' % idx]
            w = params['W%d' % idx]
            out_ch, in_ch = w.shape[:2]
            if i > 0 and i % 2 == 0:
                skip_cache = blocks['cache_skip' + str(idx-1)]
                dskip = skip_backward(dh, skip_cache)

            if i == self.L-1:
                dh = dh.reshape(*blocks['h%d' % idx].shape)

            dh, dw, db, dgamma, dbeta = conv_batchnorm_relu_backward(
                dh, h_cache)
            if not(i == 0 or in_ch == out_ch):
                # back pad trick
                dh = dh[:, :, 1:, 1:]

            blocks['dbeta%d' % idx] = dbeta
            blocks['dgamma%d' % idx] = dgamma

            if i > 0 and i % 2 == 1:
                dh += dskip
            blocks['dh%d' % (idx - 1)] = dh
            blocks['dW%d' % idx] = dw
            blocks['db%d' % idx] = db

        # w gradients where we add the regulariation term
        list_dw = {key[1:]: val + self.reg * params[key[1:]]
                   for key, val in blocks.iteritems() if key[:2] == 'dW'}
        # Paramerters b
        list_db = {key[1:]: val for key, val in blocks.iteritems() if key[:2] ==
                   'db'}
        # Parameters gamma
        list_dgamma = {key[1:]: val for key, val in blocks.iteritems() if key[
            :6] == 'dgamma'}
        # Paramters beta
        list_dbeta = {key[1:]: val for key, val in blocks.iteritems() if key[
            :5] == 'dbeta'}

        grads = {}
        grads.update(list_dw)
        grads.update(list_db)
        grads.update(list_dgamma)
        grads.update(list_dbeta)

        return loss, grads
