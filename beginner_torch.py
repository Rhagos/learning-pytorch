#This is the first pytorch program, meant to be a tutorial

from __future__ import print_function
import torch

#Constructs an uninitialized 5x3 Tensor(matrix)
#x = torch.Tensor(5,3)
#Slight variances, but all values are close enough to 0
#print(x)

#Randomly initialized matrix (5x3)
x = torch.rand(5,3)


#torch.Size is a tuple

y = torch.rand(5,3)

print(x + y)
#OR torch.add(x,y)
#OR torch.add(in1,in2, out = var)
#OR y.add_(x) Replaces y
#Any operation that mutates a tensor has a _ tacked on


#standard numpy indexing like x[:,1] still works
#Converting torch tensor to numpy array
#a = torch.ones(5)
#b = a.numpy()
#print(b)
#These two (a,b) share the same memory address so changes affect both
#torch.from_numpy(numpyarray)

#Tensors can be moved to the gpu with .cuda

if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    print(x + y, "GPU")
