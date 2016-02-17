---
layout: post
title: "Training Autoencoders on ImageNet Using Torch 7"
comments: true
---
*If you are just looking for code for a convolutional autoencoder in Torch, look at this git. There are only a few dependencies, and they have been listed in requirements.sh*

## Introduction

I have recently been working on a project for unsupervised feature extraction from natural images, such as the one below: <br />  
![Heidelberg, Germany](/assets/heidelberg.jpg "Heidelberg, Germany, October 1st, 2015")<br />  

I will save the motivation for a future post. One of the methods that I was exploring at the time was [autoencoders](https://en.wikipedia.org/wiki/Autoencoder). Since most of the code in our office is written in Lua, using Torch was the logical choice. At the time, I was still learning how to create a working network architecture, so I did a lot of learning on relevant papers, such as [AEVB](http://arxiv.org/abs/1312.6114) and [AlexNet](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks). I was also looking into tutorials on how to actually write an autoencoder, for example the excellent [blog post](https://swarbrickjones.wordpress.com/2015/04/29/convolutional-autoencoders-in-pythontheanolasagne/) by Mike Swarbrick Jones.

To the best of my knowledge, there are no publicly available examples for writing autoencoders on color images. There are, however, several examples on how to write an autoencoder for the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. It might be trivial for seasoned machine learning scientists to extend the architecture from grayscale to color images, but for me it was non-trivial. The goal of this post is to provide a minimal example on how to train autoencoders on color images using Torch.

## The Big Picture
Figure X. shows the architecture of the autoencoder. The network has three convolution layers on the encoding side and three convolution layers on the decoding side. Each convolution layer has a rectifier linear unit as an activation function. On the encoding side, there are two max-pooling layers after the second and third convolution layers. These pooling layers are mirrored on the decoder side for symmetry.

There are also two linear (fully connected) layer in the autoencoder. On the encoder side, the first linear layer condenses the output of the final max-pooling layer to a small set of features. On the decoder side, the second linear layer expands the output of the first linear layer back to the same size as the max-pooling layer.
