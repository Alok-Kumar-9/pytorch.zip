import torch
import numpy as np


x = torch.empty(3)              #an empty tensor like a 1-d vector
print(x)
y = torch.empty(5,3)            #an empty tensor like a 2-d vector/matrix with 5 rows and 3 columns
print(y)
z = torch.empty(5,3,2)          #an empty tensor like a 3-d vector
print(z)

a = torch.zeros(3,5)            #a tensor of 3X5 initialized with zero
print(a)

b = torch.rand(5,3)             #a tensor of 5X3 initialized with a random number between 0&1
print(b)

c = torch.ones(5,3)             #a tensor of 5X3 initialized with one
print(c)

print(c.dtype)                  #initial data-type is torch.float32

d = torch.ones(2,2, dtype=torch.int32)
print(d.dtype)
print(d)

e = torch.ones(2,3 , dtype = torch.double)              #double = float64
print(e.dtype)
print(e)
print(e.size())

#creating tensor from a list
f = torch.tensor([2.5 , 0.9, 44.5, 98,7])
print(f)
print(f.dtype)
print(f.size())

#creating tensor from a list
g = torch.tensor([2,3,4,5])
print(g)
print(g.dtype)
print(g.size())

#add operation
x = torch.rand(2,2)
y = torch.rand(2,2)
print(x)
print(y)
z = x+y
print(z)
z = torch.add(x,y)
print(z)

#in-place addition
y.add_(x)
print(y)

#subtract operation
z = x-y
print(z)

z = torch.sub(x,y)
print(z)

x.sub_(y)
print(x)

#multiply operation
z = x*y
print(z)

z = torch.mul(x,y)
print(z)

x.mul_(y)
print(x)

#similarily div for divison


#slicing
x = torch.rand(5,3)
print(x)
print(x[:,0:1])

y = torch.rand(5,3,2)
print(y)
print(y[1:,:1,:])
print(y[1,1,1])                     #prints a single element


#reshaping operation
x = torch.rand(5,3)
print(x)
y = x.view(15)
print(y)
z = x.view(3,5)
print(z)


#if we give dimensions such that the number of elements in tensor doesn't remain same, then pyTorch itselfs correct the dimensions
r = x.view(-1,5)
print(r)
print(r.size())


#converting tensor to numpy array and vice versa...

a = torch.ones(5)
print(a)
print(type(a))
b = a.numpy()
print(b)
print(type(b))

#both a and b share the same memory if run in CPU mode, hence change in reflected in the other...
a.add_(1)
print(a)
print(b)


a = np.ones(5)
print(a)
b = torch.from_numpy(a)
print(b)

a = a+1
print(a)
print(b)

print(torch.cuda.is_available())
#numpy can handle only CPU tensors, not GPU tensors


