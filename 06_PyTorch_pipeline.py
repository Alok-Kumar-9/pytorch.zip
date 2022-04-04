'''
1) Design Model(Input size, Output size, Forward Pass)
2) Construct Loss and Optimizer
3) Training Loop
    - Forward Pass: Compute Prediction loss and Optimizer
    - Backward Pass: Gradients
    - Update Weights
'''

import torch
import torch.nn as nn


#f = w.x
#f = 2.x

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)               #as we have assumed f=2*x for our this model

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction/ forward pass
def forward(x):
    return w*x

'''
#loss
#Mean Square error(MSE) in this case
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()
'''

'''
#gradient
#MSE = 1/N*((w*x-y)**2)
#dJ/dw = 1/N 2x (w*x-y)

def gradient(x, y, y_pred):
    return np.dot(2*x, y_pred-y).mean()
'''


print(f'Prediction before training = {forward(5):.3f}')

#Training
learning_rate =0.01
n_iterw = 100

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr=learning_rate)

for epoch in range(n_iterw):
    #prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    #gradient = backward pass
    l.backward()        #dl/dw

    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()

    if(epoch%10==0):
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training = {forward(5):.3f}')
