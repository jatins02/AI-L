import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets        # used to make datasets automatically from directory
import torchvision.transforms as T

import opendatasets as od
import numpy as np

# od.download("https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = T.Compose([
                    T.Resize((150, 150)),       # already resize the images to size
                    T.ToTensor(),
                    T.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
            ])

dataset_train = datasets.ImageFolder(root="intel-image-classification/seg_train/seg_train/", transform=transforms)
dataset_test = datasets.ImageFolder(root="intel-image-classification/seg_test/seg_test/", transform=transforms)

# below code occupying RAM, between the inter-conversions of arrays and datasets, since already a dataset, so directly make the dataloader
 
# Xtrain = []
# ytrain = []
# Xtest = []
# ytest = []
# for i in range(len(dataset_train)):
#     if (dataset_train[i][0].shape != torch.Size([3, 150, 150])):
#         continue
#     Xtrain.append(dataset_train[i][0])
#     ytrain.append(dataset_train[i][1])

# for i in range(len(dataset_test)):
#     if (dataset_test[i][0].shape != torch.Size([3, 150, 150])):
#         continue
#     Xtest.append(dataset_test[i][0])
#     ytest.append(dataset_test[i][1])

# X_train = np.array(Xtrain)
# X_test = np.array(Xtest)
# y_train = np.array(ytrain)
# y_test = np.array(ytest)

# print(len(X_train))         # length of dataset, 13986
# print(X_train[0].shape)     # three channels, 150 * 150

# labels 0 -> 5 correspond to the below classes
classes = {"buildings" : 0,
           "forest" : 1,
           "glacier" : 2,
           "mountain" : 3,
           "sea" : 4,
           "street": 5}


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
        self.lin2 = nn.Linear(69, 6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   # size: (batch, 8, 75, 75)
        x = self.pool2(F.relu(self.conv2(x)))   # size: (batch, 16, 25, 25)
        x = torch.flatten(x, start_dim=1)       # size: (batch, 16*25*25)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)                        # raw rawlogits returned

        return x

# not required
# class Dataset_(Dataset):
#     def __init__(self, X, y):
#         self.X = torch.tensor(X, dtype=torch.float32)
#         self.y = torch.tensor(y, dtype=torch.int64)
    
#     def __len__(self):
#         return len(self.X)
    
#     def __getitem__(self, ind):
#         return self.X[ind], self.y[ind]

# train_dataset = Dataset_(X_train, y_train)
# test_dataset = Dataset_(X_test, y_test)
        
model = CNN_classifier().to(device)
LR = 0.00069
n_epochs = 8
batchsize = 69
criterion = nn.CrossEntropyLoss().to(device)
optimiser = torch.optim.Adam(params = model.parameters(), lr=LR)

train_dataloader = DataLoader(dataset_train, batch_size=batchsize, shuffle=True)
test_dataloader = DataLoader(dataset_test, batch_size=batchsize, shuffle=True)

# training loop
for epoch in range(n_epochs):
    
    for (inputs, labels) in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)   
        predictions = model(inputs)

        loss = criterion(predictions, labels)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        print(f"epoch: {epoch+1}, loss={loss}")


# testing loop
total_accuracy = 0
total_samples = 0
with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device) 
        predictions = model(inputs)
        predictions = torch.argmax(predictions, dim=1)

        accuracy = (predictions == labels).sum().item()
        
        total_accuracy += accuracy
        total_samples += labels.size(0)

print(f"accuracy: {(total_accuracy/total_samples) * 100}")      # since X_train not anymore, so keep a counter of the number of samples to the model while testing