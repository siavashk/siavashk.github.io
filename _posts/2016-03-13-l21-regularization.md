---
layout: post
title: "Simple L2/L1 Regularization in Torch 7"
comments: true
---

## Motivation
A few days ago, I was trying to improve the performance of my trained neural networks. One popular approach is to introduce a regularization term on network parameters, so that the space of possible solutions is constrained to plausible values. One popular method is to use a p-norm, which is defined as:

$$\left \| x \right \|_p = \left (\sum^N_{i=1} \left | x_i \right |^p \right )^{1/p}$$

, where \\x\\ belongs to an \\N\\-dimensional vector space and \\i\\ indexes elements from \\x\\. Popular choices for \\p\\ are \\p=1\\ and \\p=2\\. \\p=1\\ results in the L1 norm, which is known to induce sparsity. For \\p=2\\, p-norm translates to the famous Euclidean norm. When L1/L2 regularization is properly used, networks parameters tend to stay small during training.

When I was trying to introduce L1/L2 penalization for my network, I was surprised to see that the stochastic gradient optimizer in the Torch nn package does not support regularization. 
