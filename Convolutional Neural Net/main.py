import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as T
import opendatasets as od

# od.download("https://www.kaggle.com/datasets/puneet6060/intel-image-classification")
device = "cuda" if torch.cuda.is_available() else "cpu"

transforms = T.Compose([
                    T.Resize((150, 150)),
                    T.ToTensor(),
                    T.Normalize(mean=(0.4915, 0.4823, 0.4468), std=(0.2470, 0.2435, 0.2616)),
            ])

dataset_train = datasets.ImageFolder(root="intel-image-classification/seg_train/seg_train/", transform=transforms)
dataset_test = datasets.ImageFolder(root="intel-image-classification/seg_test/seg_test/", transform=transforms)

# define the model
class CNN_classifier(nn.Module):
    def __init__(self):
        super(CNN_classifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=3)

        self.lin1 = nn.Linear(16*25*25, 69)
        self.lin2 = nn.Linear(69, 6)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))   
        x = self.pool2(F.relu(self.conv2(x)))   
        x = torch.flatten(x, start_dim=1)       
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return x
  
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

print(f"accuracy: {(total_accuracy/total_samples) * 100}")