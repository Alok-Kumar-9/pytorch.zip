import torch

#f = w.x
#f = 2.x

x = torch.tensor([1,2,3,4], dtype=torch.float32)
y = torch.tensor([2,4,6,8], dtype=torch.float32)               #as we have assumed f=2*x for our this model

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

#model prediction/ forward pass
def forward(x):
    return w*x


#loss
#Mean Square error(MSE) in this case
def loss(y, y_pred):
    return ((y_pred - y)**2).mean()

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

for epoch in range(n_iterw):
    #prediction = forward pass
    y_pred = forward(x)

    #loss
    l = loss(y, y_pred)

    #gradient = backward pass
    l.backward()        #dl/dw

    with torch.no_grad():
        w -= learning_rate * w.grad
    
    #zero gradients
    w.grad.zero_()

    if(epoch%10==0):
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')

print(f'Prediction after training = {forward(40):.3f}')
