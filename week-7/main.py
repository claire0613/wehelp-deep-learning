import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

# ----------------------------------------------------------------
# Set Random Seeds (optional but recommended)


def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may slow down training):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


# ========== Task1  ==========
class MyDataset(Dataset):
    def __init__(
        self,
        features,
        targets,
        feature_scaler,
        target_scaler,
        fit_scaler=False,
    ):
        """
        features: 特徵數據 (未標準化)
        targets: 標籤
        scaler: StandardScaler
        fit_scaler: 是否應該先 fit (只用於訓練集)
        """

        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler

        if fit_scaler:
            self.features = self.feature_scaler.fit_transform(features)
            self.targets = self.target_scaler.fit_transform(targets)
            print(f"Target Scaler Scale: {target_scaler.scale_}")
            print(f"Target Scaler Mean: {target_scaler.mean_}")
        else:
            self.features = self.feature_scaler.transform(features)
            self.targets = self.target_scaler.transform(targets)

        # 轉為 PyTorch tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32).view(-1, 1)

    def __getitem__(self, index):
        return self.features[index], self.targets[index]

    def __len__(self):
        return len(self.features)


class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 1)
        self.init_weights()

    def forward(self, features):
        features = torch.relu(self.layer1(features))
        features = torch.relu(self.layer2(features))
        features = torch.relu(self.layer3(features))
        return self.output(features)

    def he_init(self, layer):
        if isinstance(layer, nn.Linear):
            init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            if layer.bias is not None:
                init.constant_(layer.bias, 0.1)

    def init_weights(self):
        # 選擇使用 He initialization
        self.he_init(self.layer1)
        self.he_init(self.layer2)
        self.he_init(self.layer3)
        self.he_init(self.output)


def run_task1():
    print("----------- Task1 (Regression) ----------- ")
    df = pd.read_csv("gender-height-weight.csv")
    df = pd.get_dummies(df, columns=["Gender"])

    df["Gender"] = df["Gender_Male"].astype(int)
    df.drop(columns=["Gender_Female", "Gender_Male"], inplace=True)

    features = df[["Gender", "Height"]].values  # 2個特徵
    labels = df["Weight"].values.reshape(-1, 1)

    features_train, features_test, targets_train, targets_test = train_test_split(
        features, labels, test_size=0.2, random_state=1, shuffle=True
    )

    target_scaler = StandardScaler()
    feature_scaler = StandardScaler()

    task1_train_dataset = MyDataset(
        features=features_train,
        targets=targets_train,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scaler=True,
    )

    task1_test_dataset = MyDataset(
        features=features_test,
        targets=targets_test,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        fit_scaler=False,
    )

    batch_size = 32
    train_loader = DataLoader(task1_train_dataset, batch_size=batch_size, shuffle=True)

    input_size = task1_train_dataset.features.shape[1]  # 應該是 2

    model = RegressionModel(input_size)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    with torch.no_grad():
        features_test_torch = task1_test_dataset.features
        targets_test_torch = task1_test_dataset.targets

        predictions_test_z = model(features_test_torch)  # 在 z-score 空間
        mse_before = criterion(predictions_test_z, targets_test_torch).item()
        rmse_before = mse_before**0.5

        print("-----------  Before Training (Scaled) -----------")
        weight_std = target_scaler.scale_[0]
        rmse_before_lbs = rmse_before * weight_std
        print(f"Average Loss in Weight: {rmse_before_lbs:.4f} pounds")

    num_epochs = 50
    for epoch in range(num_epochs):
        for batch_features, batch_targets in train_loader:
            # forward
            batch_preds = model(batch_features)
            loss = criterion(batch_preds, batch_targets)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        features_test_torch = task1_test_dataset.features
        targets_test_torch = task1_test_dataset.targets

        predictions_test_z = model(features_test_torch)
        mse_after = criterion(predictions_test_z, targets_test_torch).item()
        rmse_after = mse_after**0.5

        print("\n-----------  After Training (Scaled) -----------")
        rmse_after_lbs = rmse_after * weight_std
        print(f"Average Loss in Weight: {rmse_after_lbs:.4f} pounds")

    print("\n")


class TitanicDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


# Model：BinaryClassifier (use BCEWithLogitsLoss)
class BinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_size, 5)
        self.layer2 = nn.Linear(5, 1)
        self.init_weights()

    def init_weights(self):
        init.xavier_uniform_(self.layer1.weight)
        init.constant_(self.layer1.bias, 0.1)
        init.kaiming_uniform_(self.layer2.weight, nonlinearity="sigmoid")
        init.constant_(self.layer2.bias, 0.1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x


def train_one_epoch(model, dataloader, criterion, optimizer):
    running_loss = 0.0

    for features, labels in dataloader:
        features, labels = features, labels

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(dataloader)
    return epoch_loss


@torch.no_grad()
def evaluate(model, dataloader):
    # model.eval()
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features, labels
        outputs = model(features)
        preds = (torch.sigmoid(outputs) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total
    return accuracy


def run_task2():
    print("----------- Task2 Binary Classification -----------")
    df = pd.read_csv("titanic.csv")
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna("None")
    df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
    df = df.join(pd.get_dummies(df["Embarked"], prefix="Embarked_"))
    df["FamilySize"] = df["Parch"] + df["SibSp"] + 1
    df["EncodeFamilySize"] = df["FamilySize"].apply(
        lambda s: 3 if 2 <= s <= 4 else (2 if s >= 5 else 1)
    )
    df.drop(["Name", "Ticket", "Cabin", "Embarked"], axis=1, inplace=True)

    features = [
        "Pclass",
        "Sex",
        "Age",
        "Fare",
        "EncodeFamilySize",
        "Embarked__C",
        "Embarked__None",
        "Embarked__Q",
        "Embarked__S",
    ]
    X = df[features].values
    y = df["Survived"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TitanicDataset(X_train_scaled, y_train)
    test_dataset = TitanicDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = BinaryClassifier(input_size=len(features))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    print("----------- Before Training -----------")
    accuracy_before = evaluate(model, test_loader)
    print(f"Test Accuracy Before Training: {accuracy_before:.2f}%")

    epochs = 50
    for epoch in range(epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        acc = evaluate(model, test_loader)

    print("----------- After Training -----------")
    accuracy_after = evaluate(model, test_loader)
    print(f"Test Accuracy After Training: {accuracy_after:.2f}%")


if __name__ == "__main__":

    # set_seed(42)
    run_task1()
    run_task2()
