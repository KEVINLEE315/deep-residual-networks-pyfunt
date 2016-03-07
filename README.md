# PyResNet
Residual Network Implementation in Python + Numpy, Inspired by Stanfors's CS321N

Implementation of ["Deep Residual Learning for Image Recognition",Kaiming
He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385)

Inspired by https://github.com/gcr/torch-residual-networks and based on my [CS321n](http://cs231n.github.io/) solutions.

This network should model the same behaviour of gcr's implementation.
Check https://github.com/gcr/torch-residual-networks for more infos about the structure.

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

Every convolution layer has a pad=1 and stride=1, except for the dimension
enhancning layers which has a stride of 2 to mantain the computational
complexity.
Optionally, there is the possibility of setting m affine layers immediatley before the softmax layer by setting the hidden_dims parameter, which should be a list of integers representing the numbe of neurons for each affine layer.

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

# Experiments

Model has 6*nSize+2 layers. 

I have implemented residual network in a similiar way of gcr's torch implementation. I am using the figure 1 left scheme from https://github.com/gcr/torch-residual-networks#cifar-effect-of-model-architecture, the only little difference is that I apply the addition after the second relu, which means that the skip path is not normalized nor Rectified. I didn't found significance improvements using the other methods so I implemented in this way to mantain the most readablity for the code (using convenience conv_norm_relu layers). 

I have trained the model for nSize = 1, 3 in the reference paper they started from nSize = 3 (20 convolution layers), so I was intrested to see how a resnet with nSize = 1	(8 convolution layers) performs in comparision with nSize = 3.

Obviously in terms of computation times, the 20 layer network porforms ~3.1 times slower

# Directory Structure
.
+-- __init__.py
+-- nnet/
+-- res_net.py
+-- train.py
+-- requirements.txt

# res_net.py

Contains the residual network model.

# train.py

Contains the main loop.

# requirements.txt

Requirements for the project.

# subdirs

Check the README.md found in all sub directories

# Requirements

	- [Python 2.7](https://www.python.org/)
	- [Cython](cython.org/)
	- [matplotlib](matplotlib.org/)
	- [numpy](www.numpy.org/)
	- [scipy](www.scipy.org/)
	- [cv2](opencv.org) (only for loading GTSRB)
	- [scikit_learn](scikit-learn.org/)

After you get Python, you can get [pip](https://pypi.python.org/pypi/pip) and install all requirements by running:
	
	pip install -r /path/to/requirements.txt


