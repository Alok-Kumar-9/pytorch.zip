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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0)prepare the data
bc = datasets.load_breast_cancer()
x,y = bc.data, bc.target
n_samples, n_features = x.shape

print(n_samples , n_features)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

#scale
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

x_train = torch.from_numpy(x_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0] , 1)
y_test = y_test.view(y_test.shape[0] , 1)


# 1)model

class LogisticRegression(nn.Module):
    def __init__(self, n_input_featues):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_featues , 1)

    def forward(self , x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

model = LogisticRegression(n_features)

# 2)loss amd optimizier
learning_rate = 0.01
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters() , lr=learning_rate)

# 3)training loop
num_epochs = 1000

for epoch in range(num_epochs):
    #forward pass & predict
    y_pred = model(x_train)
    loss = criterion(y_pred , y_train)

    #backward pass
    loss.backward()

    #update Weights
    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')


with torch.no_grad():
    y_pred = model(x_test)
    y_pred_cls = y_pred.round()
    acc = y_pred_cls.eq(y_test).sum()/float(y_test.shape[0])
    print(f'accuracy: {acc:.4f}')


