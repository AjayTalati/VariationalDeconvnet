#Variational Deconvnet

This is an attempt to train a Convolutional Network unsupervised. The network is structured in two parts, an encoder (the convolutional network) and a decoder (the deconvolutional network), and learns to recreate the input. This is based on a technique called Stochastic Gradient VB by D. Kingma and Prof. Dr. M. Welling. For which a reference implementation is provided at another [repository](https://github.com/y0ast/Variational-Autoencoder).

The network obtains similar results for the lowerbound for MNIST as described in the original [paper](http://arxiv.org/abs/1312.6114), but with a fourth of the parameters. This is done with the network described in [config.lua](https://github.com/y0ast/VariationalDeconvnet/blob/master/configs/mnist-1layer-16/config.lua)

The whole network, including our new modules, works with cutorch and runs on the GPU.

	> th main.lua -save configs/mnist-1layer-16
With CUDA:
	
	> th main.lua -cuda -save configs/mnist-1layer-16
	
Currently we are experimenting with more layers and training on a more challenging dataset.




