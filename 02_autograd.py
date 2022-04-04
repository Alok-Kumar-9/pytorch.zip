import torch

x = torch.randn(2, requires_grad=True)
print(x)

y = x+2
print(y)
z = y*y+2
print(z)
z = z.mean()
print(z)

z.backward()        #dz/dx
print(x.grad)


#what if not...will throw an error...
"""
x = torch.randn(2)
print(x)

y = x+2
print(y)
z = y*y+2
print(z)
z = z.mean()
print(z)

z.backward()        #dz/dx
print(x.grad)
"""

x = torch.randn(2, requires_grad=True)
print(x)

y = x+2
print(y)
z = y*y*2
print(z)
#z = z.mean()
#print(z)
v = torch.tensor([0.1, 1.0], dtype=torch.float32)
#in the background, it uses a vector-Jacobian product...
z.backward(v)
print(x.grad)


#x.requires_grad_(false)
#x.detach()
#with torch.no_grad():

#x.requires_grad_(False);
#print(x)

y = x.detach()
print(y)

with torch.no_grad():
    y = x+2
    print(y)


weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

weights = torch.ones(4, requires_grad=True)

for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)

    weights.grad.zero_()


