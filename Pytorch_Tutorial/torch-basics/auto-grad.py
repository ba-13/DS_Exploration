"""
Calculating Gradients with Autograd
"""
#%%
import torch

#%%
x = torch.randn(3)
print(x)

# %%
x = torch.randn(3, requires_grad=True)
print(x)
# %%
y = x + 2
# now as requires_grad is True, Pytorch will keep a record of a computational graph of the operations
print(y)
# tensor([1.3950, 2.9167, 2.4415], grad_fn=<AddBackward0>)
# %%
z = y * y * 2
print(z)
# %%
z = z.mean()
print(z)
# %%
z.backward()
print("grad of z wrt x: ", x.grad)
# %%
# print("grad of z wrt y: ", y.grad) # this won't work as y isn't a leaf node
rand = torch.rand(4, requires_grad=True)
rand_ = rand @ rand
print(rand)
print(rand_)
# %%
rand_.backward()  # this value must be a scalar to do .backward() without an argument
print(rand.grad)
# %%
print("Understanding grads")
x = torch.tensor([3.4, 1.2, 9.0], requires_grad=True)
print(x)
y = x * 2
print(y)
# v = torch.tensor([1, 0, 0], dtype=torch.float32)
# v = torch.tensor([0, 1, 0], dtype=torch.float32)
# v = torch.tensor([0, 0, 1], dtype=torch.float32)
v = torch.tensor([1, 1, 10], dtype=torch.float32)
y.backward(v)
print(x.grad)
# %%
# * To remove gradient calculation requirement from a tensor
# x.requires_grad_()
# x.detach()
# with torch.no_grad():
x = torch.tensor([3.4, 1.2, 9.0], requires_grad=True)
print(x)
x.requires_grad_(False)
print(x)
# %%
x = torch.tensor([3.4, 1.2, 9.0], requires_grad=True)
print(x)
x.detach_()
print(x)
# %%
x = torch.tensor([3.4, 1.2, 9.0], requires_grad=True)
print(x)
y = x + 2
print(y)
with torch.no_grad():
    y = x + 2
    print(y)
# %%
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_out = (weights * 3).sum()
    model_out.backward()
    print(weights.grad)
    # grads always get added up, which we don't want
# %%
weights.grad.zero_()
for epoch in range(3):
    model_out = (weights * 3).sum()
    model_out.backward()
    print(weights.grad)
    weights.grad.zero_()
# %%
# weights = torch.ones(4, requires_grad=True)
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
