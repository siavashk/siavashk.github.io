---
layout: post
title: "Simple L2/L1 Regularization in Torch 7"
comments: true
---

## Motivation
A few days ago, I was trying to improve the generalization ability of my neural networks. One popular approach to improve performance is to introduce a regularization term during training on network parameters, so that the space of possible solutions is constrained to plausible values. One popular method is to use a p-norm, which is defined as:

$$\left \| x \right \|_p = \left (\sum^N_{i=1} \left | x_i \right |^p \right )^{1/p}$$

, where \\(x\\) belongs to an \\(N\\)-dimensional vector space and \\(i\\) indexes elements from \\(x\\). Popular choices for \\(p\\) are \\(p=1\\) and \\(p=2\\). \\(p=1\\) results in the L1 norm, which is known to induce sparsity. For \\(p=2\\), p-norm translates to the famous Euclidean norm. When L1/L2 regularization is properly used, networks parameters tend to stay small during training.

When I was trying to introduce L1/L2 penalization for my network, I was surprised to see that the stochastic gradient descent (SGDC) optimizer in the Torch nn package does not support regularization out-of-the-box. Thankfully, you can easily add regularization using the callback.

## Adding Regularization to SGDC
Torch's [implementation](https://github.com/torch/nn/blob/master/StochasticGradient.lua) of SGDC is simple to follow. The relevant part of the optimizer is the following three lines of code:

```
currentError = currentError + criterion:forward(module:forward(input), target)
module:updateGradInput(input, criterion:updateGradInput(module.output, target))
module:accUpdateGradParameters(input, criterion.gradInput, currentLearningRate)
```

The first line calculates the loss using the forward pass of the network (`module`) given the input and current network parameters. The second line calculates the gradient of the model with respect to parameters. The third line updates network parameters using the `currentLearningRate`. In order to add regularization, we need to modify the `currentError` to reflect L1/L2 regularization penalty and also modify the update rule for network parameters. This can be achieved using the following callback function in SGDC:

```
if self.hookIteration then
  self.hookIteration(self, iteration, currentError)
end
```

If the `hookIteration` function is defined and passed to SGDC, it is called at every iteration. We can define a suitable `callback` function to implement regularization and pass it onto SGDC. One implementation can be the following:

```
local function callback(trainer, iteration, currentError)
  currentError = currentError + regularization_penalty(trainer.module, l1_weight, l2_weight)
  regularize_parameters(trainer.module, l1_weight, l2_weight)
end
```

where `trainer` is an instance of SGDC. We can add our callback to SGDC by overriding the `trainer.hookIteration = callback` function of SGDC. `currentError` is a reference to the trainer `currentError` so updating it also updates current optimizer error.

Now, creating `regularization_penalty` and `regularize_parameters` functions is easy. The first one is not strictly necessary, given that we analytically know how to differentiate L1/L2 norms, but it might be useful to implement them so that we can visualize the total loss during optimization. This can be achieved in the following manner:

```
function regularization_penalty(network, l1_weight, l2_weight)
  local parameters, _ = network:parameters()
  local penalty = 0
  for i=1, table.getn(parameters) do
    penalty = penalty + l1_weight * parameters[i]:norm(1) + l2_weight * parameters[i]:norm(2)
  end
  return penalty
end
```

The only ambiguous line might be the iteration over parameters. This is actually not an iteration over individual network parameters, but over network layers, i.e. `i` goes from 1 to number-of-layers. Now, lets' move to updating network parameters:

```
function regularize_params(network, l1_weight, l2_weight)
local parameters, _ = network:parameters()
  for i=1, table.getn(parameters) do
    local update = torch.clamp(parameters[i], -l1_weight, l1_weight)
    update:add(parameters[i]:mul(-options.l2_weight))
    parameters[i]:csub(update)
  end
end
```

`parameters` is a reference network parameters, so updating it affects the state of the network. The only ambiguous part is the `clamp` function on the parameters to the `l1_weight`. By doing this we are effectively reducing the step size when `parameters` are close to zero, thereby reducing oscillatory movement around the origin.
