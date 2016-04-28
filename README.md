## PyResNet

Residual Network Implementation in [PyFunt](https://github.com/dnlcrl/PyFunt) (a simple Python + Numpy DL framework)

Implementation of ["Deep Residual Learning for Image Recognition", Kaiming
He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385)

Also inspired by [this implementation in Lua + Torch](https://github.com/gcr/torch-residual-networks).

The network operates on minibatches of data that have shape (N, C, H, W)
consisting of N images, each with height H and width W and with C input
channels. It has, like in the reference paper, (6*n)+2 layers,
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


## Requirements

- [Python 2.7](https://www.python.org/)
- [numpy](www.numpy.org/)
- [pyfunt](https://github.com/dnlcrl/PyFunt)
- [pydatset](https://github.com/dnlcrl/PyDatSet)


After you get Python, you can get [pip](https://pypi.python.org/pypi/pip) and install all requirements by running:
	
	pip install -r requirements.txt

## Usage

If you want to train the network on the CIFAR-10 dataset, simply run:

	python train.py --help
	
Otherwise, you have to get the right train.py for MNIST or SFDDD datasets, they are respectively on the mnist and sfddd git branches:

	- train.py for MNIST: https://github.com/dnlcrl/PyResNet/blob/mnist/train.py
	- train.py for SFDDD: https://github.com/dnlcrl/PyResNet/blob/sfddd/train.py

## Experiments Results

You can view all the experiments results in the [./docs directory](https://github.com/dnlcrl/PyResNet/tree/master/docs). Main results are shown below:

###  [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

best error: 9.59 % (accuracy: 0.9041) with a 20 layers residual network (n=3):

[![CIFAR-10 results](https://github.com/dnlcrl/PyResNet/blob/master/docs/imgs/cifar.png)](https://github.com/dnlcrl/PyResNet/blob/master/docs/CIFAR-10%20Experiments.ipynb)

###  [MNIST](http://yann.lecun.com/exdb/mnist/)

best error: 0.36 % (accuracy: 0.9964) with a 32 layers residual network (n=5):

[![MNIST results](https://github.com/dnlcrl/PyResNet/blob/master/docs/imgs/mnistres.png)](https://github.com/dnlcrl/PyResNet/blob/master/docs/MNIST%20Experiments.ipynb)

###  [SFDDD](https://www.kaggle.com/c/state-farm-distracted-driver-detection)

best error: 0.25 % (accuracy: 0.9975 %) on a subset (1000 samples) of the train data (~21k images) with a 44 layers residual network (n=7), resizing the images to 64x48, randomly cropping 32x32 images for training and cropping a 32x32 image from the center of the original images for testing. Unfortunately I got more than 2% error on Kaggle's results (composed of ~80k images).
	
WIP

## TODOs:

- regenerate plots with english labels

- experimentat elastic distortion as data augmentation funcion on MNIST

- experiment other data augmentation funcions on SFDDD

- implementation of the second version of residual networks, as explained in ["Identity Mappings in Deep Residual Networks" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun](http://arxiv.org/pdf/1603.05027v1.pdf) 
