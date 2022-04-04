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

x = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)
y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)               #as we have assumed f=2*x for our this model
n_samples, n_features = x.shape
print(n_samples, n_features)


'''
w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction/ forward pass
def forward(x):
    return w*x
'''

input_size = n_features
output_size = n_features

#model = nn.Linear(input_size , output_size)

class LinearRegression(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim , output_dim)

    def forward(self, x):
        return self.lin(x)

model = LinearRegression(input_size, output_size)


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

x_test = torch.tensor([5], dtype=torch.float32)
print(f'Prediction before training = {model(x_test).item():.3f}')

#Training
learning_rate =0.01
n_iterw = 1000

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iterw):
    #prediction = forward pass
    y_pred = model(x)

    #loss
    l = loss(y, y_pred)

    #gradient = backward pass
    l.backward()        #dl/dw

    #update weights
    optimizer.step()
    
    #zero gradients
    optimizer.zero_grad()

    if(epoch%10==0):
        [w,b] = model.parameters()
        print(f'epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}')

print(f'Prediction after training = {model(x_test).item():.3f}')
