import torch
import torch.nn as nn
import numpy as np

def cross_entropy(actual , predicted):
    loss = -1*np.sum(actual * np.log(predicted))
    return loss

#y must be one hot encoded
#if class 0: [1 0 0]
#if class 1: [0 1 0]
#if class 2: [0 0 1]

y = np.array([1,0,0])

#y_pred has probabilities
y_pred_good = np.array([0.7, 0.2, 0.1])
y_pred_bad = np.array([0.1, 0.3, 0.6])

l1 = cross_entropy(y, y_pred_good)
l2 = cross_entropy(y, y_pred_bad)

print(f'Loss1 : {l1:.4f}')
print(f'Loss2 : {l2:.4f}')


loss = nn.CrossEntropyLoss()

y = torch.tensor([0])
#n_samples * n_classes = 1*3

y_pred_good = torch.tensor([[2.0 , 1.0 , 0.1]])
y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(y_pred_good , y)
l2 = loss(y_pred_bad , y)

print(l1.item())
print(l2.item())

_,predictions1 = torch.max(y_pred_good , 1)
_,prediction2 = torch.max(y_pred_bad , 1)

print(predictions1 , prediction2)


Y = torch.tensor([2,0,1])
#n_samples * n_classes = 3*3

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1],[2.0, 1.0, 0.1],[0.1, 3.0, 0.6]])
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1],[0.1, 1.0, 2.0],[0.1, 3.0, 0.1]])

l1 = loss(Y_pred_good , Y)
l2 = loss(Y_pred_bad , Y)

print(l1.item())
print(l2.item())

_,predictions1 = torch.max(Y_pred_good , 1)
_,prediction2 = torch.max(Y_pred_bad , 1)

print(predictions1 , prediction2)


