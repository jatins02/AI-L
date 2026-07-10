import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets        # used to make datasets automatically from directory
import torchvision.transforms as T

import random
import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# od.download("https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = T.Compose([
                    T.ToTensor(),
                    T.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
            ])

dataset = datasets.ImageFolder(root="intel-image-classification/seg_test/seg_test/", transform=transforms)

it = iter(dataset)
inputs, labels = next(it)       # input is an image in three channels and labels is an int label

X = []
y = []
for i in range(len(dataset)):
    if (dataset[i][0].shape != torch.Size([3, 150, 150])):
        continue
    X.append(dataset[i][0])
    y.append(dataset[i][1])

combined = list(zip(X, y))
random.shuffle(combined)

X, y = zip(*combined)
X = list(X)
y = list(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# print(len(X_train))         # length of dataset, 1885
# print(X_train[0].shape)     # three channels, 150 * 150

# labels 0 -> 4 correspond to the below classes
classes = {"buildings" : 0,
           "forest" : 1,
           "glacier" : 2,
           "mountain" : 3,
           "sea" : 4}


# define the model
class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()

        # to keep the size of input and output the same, padding = (kernel_size - 1) // 2

        # first convolutional layer, and its pooling layer
        # input: (batch, 3, 150, 150), output: (batch, 8, 150, 150)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        # downsample by half
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # second convolutional layer
        # input: (batch, 8, 75, 75), output: (batch, 16, 25, 25)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        # total flattened features (16 channels * 25 * 25)
        self.lin1 = nn.Linear(16*25*25, 69)
        self.lin2 = nn.Linear(69, 5)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # size: (batch, 8, 75, 75)
        x = self.pool2(F.relu(self.conv2(x)))   # size: (batch, 16, 25, 25)
        x = torch.flatten(x, start_dim=1)       # size: (batch, 16*25*25)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)                        # raw rawlogits returned

        return x
    
class Dataset_(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).to(device)
        self.y = torch.tensor(y, dtype=torch.int64).to(device)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, ind):
        return self.X[ind], self.y[ind]

print(type(X_train))

train_dataset = Dataset_(X_train, y_train)
test_dataset = Dataset_(X_test, y_test)
        
model = CNN_classifier().to(device)
LR = 0.0069
n_epochs = 44
batchsize = 5
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(params = model.parameters(), lr=LR)

train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

# training loop
for epoch in range(n_epochs):
    
    for (inputs, labels) in train_dataloader:   
        predictions = model(inputs)

        loss = criterion(predictions, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        print(f"epoch: {epoch+1}, loss={loss}")


# testing loop
with torch.no_grad():
    total_accuracy = 0
    for inputs, labels in test_dataloader:
        predictions = model(inputs)     # 5*5 matrix, with probabilities of each label
        predictions = torch.argmax(predictions, dim=1)

        accuracy = (predictions == labels).sum().item()
        
        total_accuracy += accuracy

print(f"accuracy: {(total_accuracy/len(X_test)) * 100}")