'''
Activation Functions - Used to decide whether to activate a neuron or not.
Used in between 2 layers of neurons
1) Step Function - 1 for +ve values & 0 for -ve values. Not generally used in practice.
2) Sigmoid Function - 1/(1+(e^(-x))). Used generally for a binary classification problems.
3) TanH Function - 2/(1+(e^(-2x)))-1. Scaled and shifted sigmoid fuction.
4) ReLU Function - max(0,x). Best choice for Neural Network. If you don't know what to use, use ReLU.
5) Leaky ReLU - x for x>=0 & ax for x<0. Tries to solve the vanishing gradient problem.
6) Softmax - Used in multi-class classification problems. Squashes the inputs to give outputs b/t 0&1.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

#Option 1 - Use nn Modules
class NeuralNet1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        #nn.Softmax
        #nn.LeakyReLU
        #nn.TanH
        #nn.Sigmoid
        self.linear2 = nn.Linear(hidden_size, 1)            #as it is binary classification, output_size/num_classes=1
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out


#Option 2 - use activation Functions directly instead
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        #self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)            #as it is binary classification, output_size/num_classes=1
        #self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #F.leaky_relu(correct)
        #F.relu(also correct)
        #torch.relu
        #torch.tanh
        #torch.leakyReLu(Xwrong.. can't be used)
        #torch.sigmoid
        #torch.softmax
        out = torch.relu(self.linear1(x))
        out = torch.sigmoid(self.linear2(out))
        return out




