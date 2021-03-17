---
layout: post
title: "Named Tensors in PyTorch"
comments: true
---

## Problem
A couple of weeks ago, I was debugging a training script for a semantic segmentation problem.
There were no run-time errors, but the loss was not decreasing:

```python
...
loss = criterion(predictions, labels) # criterion is an instance of BCEWithLogitsLoss.
...
```

Upon inspection, I found that the issue was caused by an inconsistency between these lines:

```python
for images, labels in loader: # labels has shape (N, W, H)
  predictions = model.forward(images) # predictions has shape (N, H, W)
  ...
```

Because `W = H = 256` in my dataset, I did not notice the issue. Now this is a simple case, and a proper unit test would have found this.
But suppose we had a more complicated example:

```python
batch = next(iter(loader)) # Get a batch from a Dataloader with shape (batch=16, channels=3, height=64, width=64).
tensor = some_complicated_function(batch)
print(tensor.shape) # prints (16, 64, 64, 32, 8).
another_tensor = another_complicated_function(tensor[..., 0]) # What is 0?!
print(tensor.shape) # prints (16, 8, 32, 8). What are these dimensions?
```

It is very difficult to follow this code. In the third line, the tensor rank has increased from four to five and it is not clear what the new added dimension represents.
Furthermore, `some_complicated_function` might have permuted the second and third axes, and there is no way of knowing that from the outputs. As more and more functions are applied, the information about the tensor layout is lost even more.

## Solution
This has been a known issue for a while now. [Alexander Rush](http://nlp.seas.harvard.edu/NamedTensor) has a nice blog post about this, and thanks to him, PyTorch has implemented named tensors since 1.3. Named tensors provide a way of attaching a name to tensor dimensions that is preserved for [most operations](https://pytorch.org/docs/stable/name_inference.html). The official PyTorch [tutorial](https://pytorch.org/tutorials/intermediate/named_tensor_tutorial.html) is a great place to start if you want to try them out.

Named tensors required minimum changes to your code. For example:

```python
tensor = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))
print(tensor.shape) # prints torch.Size([1, 2, 2, 3]).
print(tensor.names) # prints ('N', 'C', 'H', 'W').
```

If you reduce one of the dimensions, the resulting tensor would have appropriate names:

```python
tensor = torch.randn(1, 2, 2, 3, names=('N', 'C', 'H', 'W'))
reduced = torch.sum(tensor, axis=2)
print(reduced.names) # prints ('N', 'C', 'W')
```

There are a lot of cool use-cases in the official documentation. So be sure to check them out and hopefully they will be useful for your development.
