## PyResNet

Residual Network Implementation in [PyFunt](https://github.com/dnlcrl/PyFunt) (a Python + Numpy DL framework, largely inspired by Stanfors's [CS321n](http://cs231n.github.io/))

Implementation of ["Deep Residual Learning for Image Recognition", Kaiming
He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385)

Inspired by https://github.com/gcr/torch-residual-networks.

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

I have implemented residual network to train on the [CIFAR10](link) dataset, in a similiar way of gcr's torch implementation. I essentially use the scheme in figure 1 left from [here](https://github.com/gcr/torch-residual-networks#cifar-effect-of-model-architecture), the only little difference is that I apply the addition after the second relu, which means that the skip path is not normalized nor Rectified. I didn't found significance improvements using the other methods so I implemented in this way to mantain the most readablity for the code (using convenience conv_batchnorm_relu layers). 

I have trained the model for nSize = 1, 3. In the reference paper the authors start from nSize = 3 (20 convolution layers), so I was intrested to see how a resnet with nSize = 1 (8 convolution layers), performs in comparision with nSize = 3.

Obviously in terms of computation times, the 20 layer network porforms ~3.1 times slower.

I all the experiments I use the following training configuration (more or less the same as the reference pa per):

- update rule: SGD with nesterov momentum (= 0.9) ;
- weight decay: 1e-4;
- learning rate: 0.1 for 80 epochs, 0.01 for the next 40 epochs, 0.001;
- epochs: 160~200

batch size varies from 50 to 128 and processes from 1 to 8, but they should not influence the final accuracy.

## Effects of Network Size


## Effects of Data Augmentation

I also wanted to see the difference in accuracy and loss when we augment the dataset by adding n white pixels on each side of each image (and random cropping 32x32 images before each step), with n euqal to 2 and 4 (the authors use n=4), for both nSize = 1, 3.

In all cases I also applied random mirroring like gcr's implementation.

## Weight Visualization

## Images not recognized

## Fooling the Network

## Directory Structure
	.
	+-- __init__.py
	+-- nnet/
	+-- res_net.py
	+-- train.py
	+-- requirements.txt

### res_net.py

Contains the residual network model.

### train.py

Contains the main loop.

### requirements.txt

Requirements for the project.

## subdirs

Check the README.md found in all sub directories

## Requirements

- [Python 2.7](https://www.python.org/)
- [Cython](cython.org/)
- [matplotlib](matplotlib.org/)
- [numpy](www.numpy.org/)
- [scipy](www.scipy.org/)
- [cv2](opencv.org) (only for loading GTSRB)
- [scikit_learn](scikit-learn.org/)


After you get Python, you can get [pip](https://pypi.python.org/pypi/pip) and install all requirements by running:
	
	pip install -r requirements.txt


