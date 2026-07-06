import torch
import torch.nn as nn
import torch.nn.functional as F
import opendatasets as od
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# od.download("https://www.kaggle.com/datasets/yasserh/wine-quality-dataset/data")

device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("wine-quality-dataset/WineQT.csv")
df = df.drop(["Id"], axis=1)
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
# df shape: (1018, 12)


# normalise the columns of dataset
df_cols = df.columns
for i in range(len(df_cols) - 1):
    col = df_cols[i]
    df[col] /= df[col].abs().max()


# split dataframe into testing and training (stratify the dataset, based on "quality" column)
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True, stratify=df["quality"])


# split the dataset furthur into inputs and outputs
x_train = df_train.iloc[:, :-1]
y_train = df_train.iloc[:, -1]

x_test = df_test.iloc[:, :-1]
y_test = df_test.iloc[:, -1]


# since y_train and y_test do not have zero indexed labels, so make them zero indexed
min_quality_label = y_train.min()
y_train_adjusted = y_train - min_quality_label
y_test_adjusted = y_test - min_quality_label


# define the model
class LinearRegression(nn.Module):
    def __init__(self, input_features):
        super(LinearRegression, self).__init__()
        self.lin1 = nn.Linear(input_features, 6)

    def forward(self, x):
        return self.lin1(x)


# setup for the training loop
model = LinearRegression(x_train.shape[1]).to(device)
LR = 0.22
optimiser = torch.optim.SGD(model.parameters(), lr=LR)
iters = 8
criterion = nn.MSELoss()


# the training loop
for epoch in range(iters):
    for i in range(x_train.shape[0]):
        y_out = torch.zeros(6)
        ans = y_train_adjusted.iloc[i]
        y_out[ans] = 1
        y_out = y_out.to(device)
        
        input = torch.from_numpy(x_train.iloc[i, :].to_numpy().copy()).float().to(device)  ###
        y_preds = model(input)

        loss = criterion(y_preds, y_out)
        loss.backward()

        optimiser.step()
        optimiser.zero_grad()
        
        if (i % 10 == 0):
            print(f"epoch: {epoch+1}   i: {i+1}   loss: {loss}")


with torch.no_grad():
    correct = 0
    for i in range(x_test.shape[0]):
        input = torch.from_numpy(x_test.iloc[i, :].to_numpy().copy()).float().to(device)
        y_preds = model(input)
        y = torch.argmax(y_preds).item()

        if y == y_test_adjusted.iloc[i]:
            correct+=1

    print(f"accuracy: {(correct/x_test.shape[0]) * 100}")


        


