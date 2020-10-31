import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

data_train = np.random.randint(0,10,size=[2500,153])

data_test = np.random.randint(0,10,size=[300,153])

x_train = data_train
y_train = np.random.randint(0,2,size=[2500,1])
x_test = data_test


encoder_y = LabelEncoder()
encoder_y.fit(y_train)
encoded_y = encoder_y.transform(y_train)
y_train = np_utils.to_categorical(encoded_y)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(153, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()
x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)

for epoch in range(5):
    optimizer.zero_grad()
    output = model(x_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.item()))

# check predictions
output = model(x_test)
probs = torch.sigmoid(output)
print(probs)
probs = torch.softmax(output,dim=1)
print(probs)
