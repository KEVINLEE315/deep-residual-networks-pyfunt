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

Contains the main function.

### requirements.txt

Requirements for the project.

## Requirements

- [Python 2.7](https://www.python.org/)
- [numpy](www.numpy.org/)
- [pyfunt](https://github.com/dnlcrl/PyFunt)
- [pydatset](https://github.com/dnlcrl/PyDatSet)


After you get Python, you can get [pip](https://pypi.python.org/pypi/pip) and install all requirements by running:
	
	pip install -r requirements.txt

## Usage

	python train.py --help
