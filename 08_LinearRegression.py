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
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=42)

x = torch.from_numpy(x_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
y = y.view(y.shape[0],1)

n_samples, n_features = x.shape

# 1)model
input_size = n_features
output_size = 1

model = nn.Linear(input_size, output_size)


# 2)Loss and Optimizer
learning_rate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3)Training Loop
num_epoch = 1000

for epoch in range(num_epoch):
    #forward pass and Loss
    y_pred = model(x)
    loss = criterion(y_pred , y)

    #backward pass
    loss.backward()

    #update weights
    optimizer.step()
    optimizer.zero_grad()

    if (epoch+1)%10==0:
        print(f'epoch: {epoch+1}, loss: {loss.item():.4f}')

#plot
plt.figure()
predicted = model(x).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()

