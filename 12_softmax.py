import torch
import numpy as np
import torch.nn as nn

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x) , axis=0)

x = np.array([2.0, 1.0, 0.1])
outputs = softmax(x)
print(outputs)

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x , dim=0)
print(outputs)



