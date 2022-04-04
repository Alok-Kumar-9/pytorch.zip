'''
Convolutional Neural Networks(CNN) mainly work on image data.It applies convolutional filters along the optional activational functions
followed by pooling layers.At last a fully connected Neural net is created.Pooling(Max-Pooling in this case) tends to decrease the
operational cost by reducing the size of our image. CNN is mainly used for Feature Extraction.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

#device-config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyper-parameters
num_epochs = 2
batch_size = 4
learning_rate = 0.001

#dataset has PILImages of range[0,1]
#we transform them to normalized tensors ranging [-1,1]
transform = transforms.Compose([transforms.ToTensor() , transforms.Normalize((0.5, 0.5, 0.5) , (0.5,0.5,0.5))])

train_dataset = torchvision.datasets.CIFAR10(root='./', transform=transform, train=True, download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./', transform=transform, train=False, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet , self).__init()
        self.conv1 = nn.Conv2d(3 , 6 , 5)
        self.pool1 = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6 , 16 , 5)
        self.fc1 = nn.Linear(16*5*5 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)             #no activation function at last if we are using CrossEntropyLoss

        return x

model = ConvNet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for i,(images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        #Forward Pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward Pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%10==0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {i+1}/{n_total_steps}, loss: {loss.item():.4f}')
        
print('Training Finished')

with torch.no_grad():
    n_correct=0
    n_samples=0

    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        #max returns value,index
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted==labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if(label==pred):
                n_class_correct[label]+=1
            n_class_samples[label]+=1

    acc = 100*(n_correct / n_samples)
    print(f'Accuracy: {acc} %')

    for i in range(10):
        acc = 100.0 * (n_class_correct[i] / n_class_samples[i])
        print(f'Accuracy of {classes[i]}: {acc} %')


