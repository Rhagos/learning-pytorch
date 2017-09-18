import torch
from torch.autograd import Variable


x = Variable(torch.ones(2,2), requires_grad = True)
#torch.ones(2,2) = 2x2 matrix of 1s
#requires_grad = requires gradient???

print(x)

y = x + 2

print(y)

print(y.grad_fn)

z = y * y * 3
out = z.mean()

print(z, out)

#Backpropagation

#out.backward() = out.backward(torch,Tensor([1.0]))
out.backward()
#print gradients d(o)/dx
print(x.grad)


x = torch.randn(3)
x = Variable(x, requires_grad = True)

y = x*2
while y.data.norm() < 1000:
    y = y*2

print(y)

