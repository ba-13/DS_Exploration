import torch
import numpy as np

x = torch.empty(1)
x = torch.empty(3)
x = torch.empty(2, 3)
x = torch.empty(4, 3, 2)

x = torch.rand(2, 3)
x = torch.randn(2, 3)

x = torch.zeros(2, 2)
x = torch.ones(2, 2)
x = torch.ones(2, 2, dtype=torch.int)

x = torch.tensor([2, 324, 53, 69.0])

x = torch.rand(2, 2)
y = torch.randn(2, 2)
z = x + y

z = torch.add(x, y)

y.add_(x)
# every _ ending function does an inplace operation

z = y - x
z = torch.sub(y, x)

z = y * x
z = torch.mul(y, x)

x = torch.rand(5, 3)
print("x is ", x)
print(x[:, 0])
print(x[:, 0].size())

x = torch.rand(4, 5)
print(x[3, 2])
print(x[3, 2].item())  # item() can be used for only scalar values

x = torch.rand(4, 4)
print(x)
print(x.size())
y = x.view(-1, 2)  # pytorch decides the -1 dimension automatically
print(y)
print(y.size())

a = torch.ones(3, 2)
print(a)
b = a.numpy()
print(b)
# if the tensor is on CPU, then both objects share the same memory location, so changing one changes the other

a = np.ones((2, 6))
b = torch.from_numpy(a)
print(b)

if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.ones(5, device=device)
    y = torch.ones(5)
    y = y.to(device)
    z = x + y
    print(z)

    z = z.to("cpu")
    # to_numpy won't work on gpu variables

# if you want to specify that you might need to calculate gradients of the tensor, then for optimization, do the following:
x = torch.ones(4, requires_grad=True)
print(x)
