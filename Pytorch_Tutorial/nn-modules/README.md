Without activation functions, our network essentially behaves like a single linear transformation, a stacked linear regression model, which doesn't fit more complex decision spaces
Tanh is a good choice in the hidden layers.
When weights are dead due to vanishing gradients, i.e. they aren't updating, try using leaky ReLU instead.

Specific Activation functions can be called from `torch.nn.functional`
