import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# 读取train.csv和test.csv
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 删除id和cityCode列
train_data = train_data.drop(['id', 'cityCode'], axis=1)
test_data = test_data.drop(['id', 'cityCode'], axis=1)


# 定义房价预测数据集类
class HousePriceDataset(Dataset):
    def __init__(self, data):
        self.features = torch.tensor(data.iloc[:, 1:].values, dtype=torch.float32)
        self.labels = torch.tensor(data.iloc[:, 0].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# 将train数据集拆分成训练集和验证集
train_set, val_set = train_test_split(train_data, test_size=0.25)

# 创建数据集实例
train_dataset = HousePriceDataset(train_set)
val_dataset = HousePriceDataset(val_set)
test_dataset = HousePriceDataset(test_data)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 定义神经网络模型
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x


# 初始化模型
model = HousePriceModel()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 定义early stopping
best_loss = float('inf')
patience = 3
num_epochs = 100
counter = 0

# 训练模型
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 在验证集上计算loss并更新模型
    val_loss = 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs,
