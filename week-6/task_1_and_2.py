import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from random import uniform

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class Layer:
    weight: list
    bias_weight: list


class ActivationFunctions:
    @classmethod
    def relu(cls, x):
        return x if x > 0 else 0

    @classmethod
    def linear(cls, x):
        return x

    @classmethod
    def sigmoid(cls, x):
        return 1 / (1 + math.exp(-x))

    @classmethod
    def softmax(cls, values):
        # Subtract max(values) for numerical stability
        max_value = max(values)
        exp_values = [math.exp(v - max_value) for v in values]
        sum_exp = sum(exp_values)
        return [v / sum_exp for v in exp_values]

    @classmethod
    def linear_derivative(cls, x):
        return 1

    @classmethod
    def relu_derivative(cls, x):
        # ReLU 函數的導數，若 x > 0 則為 1，否則為 0
        return 1 if x > 0 else 0

    @classmethod
    def sigmoid_derivative(cls, output):

        return output * (1 - output)


class LossFunctionHelper:
    @classmethod
    def calculate_mse(cls, expected_values, output_values):
        n = len(expected_values)
        squared_errors = 0

        for i in range(n):
            squared_errors += (expected_values[i] - output_values[i]) ** 2

        mse = squared_errors / n
        return mse

    @classmethod
    def binary_cross_entropy(cls, expected_values, output_values):

        n = len(expected_values)
        loss = 0.0
        for i in range(n):
            output_value = max(min(output_values[i], 1 - 1e-7), 1e-7)
            loss += expected_values[i] * math.log(output_value) + (
                1 - expected_values[i]
            ) * math.log(1 - output_value)
        return -loss

    @classmethod
    def bce_derivative(cls, expected_values, output_values):
        # 二元交叉熵 (BCE) 的導數：- (E / O) + ((1 - E) / (1 - O))

        return [
            -(E_i / max(O_i, 1e-7)) + ((1 - E_i) / max(1 - O_i, 1e-7))
            for E_i, O_i in zip(expected_values, output_values)
        ]

    @classmethod
    def mse_derivative(cls, expected_values, output_values):
        # 均方誤差 (MSE) 的導數：2/n * (O - E)
        n = len(output_values)  # 輸出大小
        return [
            (2 / n) * (O_i - E_i) for O_i, E_i in zip(output_values, expected_values)
        ]

    @classmethod
    def compute_loss(
        cls,
        expected_values: list[float],
        output_values: list[float],
        loss_type: str = "mse",
    ) -> float:
        """動態計算損失 (可選 MSE 或 BCE)"""
        if loss_type == "mse":
            return cls.calculate_mse(expected_values, output_values)
        elif loss_type == "bce":
            return cls.binary_cross_entropy(expected_values, output_values)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @classmethod
    def compute_derivative(
        cls,
        expected_values: list[float],
        output_values: list[float],
        loss_type: str = "mse",
    ) -> list[float]:
        """動態計算損失函數的導數"""
        if loss_type == "mse":
            return cls.mse_derivative(expected_values, output_values)
        elif loss_type == "bce":
            return cls.bce_derivative(expected_values, output_values)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")


class Network(ABC):
    def __init__(self, bias: float, layers: list[Layer]):
        self.bias = bias
        self.layers = layers
        self.output_data = None
        self.layers_output = []
        self.input_date = []

        self.gradient = None

    @abstractmethod
    def activation_function(self, input: float, layer_num: int):
        pass

    def forward(self, input_data: list):
        current_input = input_data
        self.input_data = input_data
        self.layers_output = []  # **確保 layers_output 重新初始化**

        for i in range(len(self.layers)):
            # 計算每層的內積 + 偏置
            layer_output = []
            for neuron_weight, bias_weight in zip(
                self.layers[i].weight, self.layers[i].bias_weight
            ):

                element = (
                    sum(
                        weight * input_data
                        for weight, input_data in zip(neuron_weight, current_input)
                    )
                    + bias_weight * self.bias
                )
                # print(f"Layer {i} Output Before Activation: {element}")  # **檢查數值範圍**
                element = self.activation_function(element, i)
                layer_output.append(element)

            current_input = layer_output
            self.layers_output.append(current_input)

        return current_input

    def zero_grad(self, learning_rate):
        if self.gradient is None:
            return

        for layer_idx, layer in enumerate(self.layers):
            input_data = (
                self.layers_output[layer_idx - 1] if layer_idx > 0 else self.input_data
            )
            # 檢查 gradient 和 input_data 長度
            for i in range(len(layer.weight)):
                for j in range(len(layer.weight[i])):
                    # 確保 gradient 的長度和 input_data 的長度一致
                    if j >= len(input_data):
                        continue
                    weight_gradient = self.gradient[layer_idx][i] * input_data[j]

                    layer.weight[i][j] -= learning_rate * weight_gradient

                # 更新 bias 權重
                bias_gradient = self.gradient[layer_idx][i] * self.bias
                layer.bias_weight[i] -= learning_rate * bias_gradient
        self.gradient = None

    @abstractmethod
    def backward(self, output_losses):
        pass


# He 初始化 (適用於 ReLU 層)
def he_init(fan_in, fan_out):
    limit = math.sqrt(2 / fan_in)
    return [[uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]


# Xavier 初始化 (適用於 Sigmoid 層)
def xavier_init(fan_in, fan_out):
    limit = math.sqrt(6 / (fan_in + fan_out))
    return [[uniform(-limit, limit) for _ in range(fan_in)] for _ in range(fan_out)]


class Regression(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return ActivationFunctions.linear(input)

    def backward(self, loss_gradient: list[float]):
        layer_gradients = [[] for _ in range(len(self.layers))]
        # 計算輸出層的梯度
        output_layer_idx = len(self.layers) - 1
        output_layer = self.layers[output_layer_idx]

        # 輸出層的梯度: 直接由 loss_gradient 和 activation_derivative 計算
        activation_derivatives = [
            ActivationFunctions.linear_derivative(x)
            for x in self.layers_output[output_layer_idx]
        ]

        layer_gradients[output_layer_idx] = [
            loss_gradient[i] * activation_derivatives[i]
            for i in range(len(output_layer.weight))
        ]

        # 計算隱藏層的梯度（從後往前）
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            # 計算當前層的激活函數的導數
            activation_derivatives = [
                (
                    ActivationFunctions.relu_derivative(x)
                    if layer_idx == 0
                    else ActivationFunctions.linear_derivative(x)
                )
                for x in self.layers_output[layer_idx]
            ]

            # 計算當前層的梯度
            layer_gradients[layer_idx] = [
                sum(
                    layer_gradients[layer_idx + 1][j] * next_layer.weight[j][i]
                    for j in range(len(next_layer.weight))
                )
                * activation_derivatives[i]
                for i in range(len(layer.weight))
            ]

        # 儲存梯度
        self.gradient = layer_gradients


class BinaryClassification(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return ActivationFunctions.sigmoid(input)

    def backward(self, loss_gradient: list[float]):
        layer_gradients = [[] for _ in range(len(self.layers))]

        # 計算輸出層的梯度
        output_layer_idx = len(self.layers) - 1
        output_layer = self.layers[output_layer_idx]
        activation_derivatives = [
            ActivationFunctions.sigmoid_derivative(x)
            for x in self.layers_output[output_layer_idx]
        ]

        layer_gradients[output_layer_idx] = [
            loss_gradient[i] * activation_derivatives[i]
            for i in range(len(output_layer.weight))
        ]

        # # 計算隱藏層的梯度（從後往前）
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]
            activation_derivatives = [
                (
                    ActivationFunctions.relu_derivative(x)
                    if layer_idx == 0
                    else ActivationFunctions.sigmoid_derivative(x)
                )
                for x in self.layers_output[layer_idx]
            ]

            layer_gradients[layer_idx] = [
                sum(
                    layer_gradients[layer_idx + 1][j] * next_layer.weight[j][i]
                    for j in range(len(next_layer.weight))
                )
                * activation_derivatives[i]
                for i in range(len(layer.weight))
            ]

        # 儲存梯度
        self.gradient = layer_gradients


def transform_loss_to_weights(loss, weight_std):
    loss = loss * (weight_std**2)
    return math.sqrt(loss)


print("----------- Task1 ----------- ")

# ---------- 初始化神經網絡 ----------
nn = Regression(
    bias=1,
    layers=[
        Layer(
            weight=he_init(2, 2),
            bias_weight=[0.1] * 2,
        ),
        Layer(
            weight=xavier_init(2, 1),
            bias_weight=[0.1],
        ),
    ],
)

# ---------- 讀取 & 預處理數據 ----------
df = pd.read_csv("gender-height-weight.csv")
df = pd.get_dummies(df, columns=["Gender"])
df["Gender"] = df["Gender_Male"].astype(int)  # 使用單一數字表示性別
df.drop(columns=["Gender_Female", "Gender_Male"], inplace=True)  # 移除多餘欄位
# 標準化數據
weight_std = df["Weight"].std()
weight_mean = df["Weight"].mean()
df["Height"] = (df["Height"] - df["Height"].mean()) / df["Height"].std()
df["Weight"] = (df["Weight"] - weight_mean) / weight_std
# 定義 X, y
X = df[["Gender", "Height"]].values.tolist()  # 2個特徵
y = df["Weight"].to_numpy().reshape(-1, 1)

# ---------- 訓練神經網絡 ----------
epochs = 30
learning_rate = 0.01
loss_func = LossFunctionHelper()
print(" -----------  Before  Training -----------")
total_loss_before = 0
for i in range(len(X)):
    inputs = X[i]
    expected = y[i]
    outputs = nn.forward(inputs)
    loss = loss_func.calculate_mse(expected, outputs)
    total_loss_before += loss


average_loss_before = total_loss_before / len(X)
print(
    f"Average Loss in Weight: {transform_loss_to_weights(average_loss_before,weight_std)} pounds"
)
print(" -----------  After  Training -----------")
for epoch in range(epochs):
    total_loss = 0
    for i in range(len(X)):
        inputs = X[i]
        expected = y[i]
        outputs = nn.forward(inputs)
        loss = loss_func.calculate_mse(expected, outputs)
        total_loss += loss

        output_gradient = loss_func.mse_derivative(expected, outputs)
        nn.backward(output_gradient)
        nn.zero_grad(learning_rate)
    average_loss_after = total_loss / len(X)
print(
    f"Average Loss in Weight: {transform_loss_to_weights(average_loss_after,weight_std)} pounds"
)


print(" ")
print("----------- Task2 ----------- ")


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


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 確保 X_train 是 NumPy 陣列
X_train = np.asarray(X_train, dtype=np.float64)
X_test = np.asarray(X_test, dtype=np.float64)

# 計算均值和標準差
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)

# 標準化
X_train = (X_train - X_mean) / X_std
X_test = (X_test - X_mean) / X_std


nn = BinaryClassification(
    bias=1,
    layers=[
        Layer(weight=xavier_init(9, 5), bias_weight=[0.1] * 5),
        Layer(weight=he_init(5, 1), bias_weight=[0.1]),
    ],
)

learning_rate = 0.01

epochs = 30
learning_rate = 0.01
loss_func = LossFunctionHelper()
print(" -----------  Before  Training -----------")
# Evaluating Procedure
threshold = 0.5


def evaluate_model(nn, X, y):
    correct_count = sum((nn.forward(x)[0] >= threshold) == e for x, e in zip(X, y))
    return (correct_count / len(y)) * 100

print(f"correct_rate : {evaluate_model(nn, X_test, y_test)} %")
for epoch in range(epochs):
    total_loss = 0
    for x, e in zip(X_train, y_train):

        outputs = nn.forward(x)
        loss = loss_func.binary_cross_entropy([e], outputs)
        total_loss += loss
        loss_gradient = LossFunctionHelper.compute_derivative([e], outputs, "bce")
        nn.backward(loss_gradient)
        nn.zero_grad(learning_rate)

print(" -----------  After  Training -----------")


print(f"correct_rate : {evaluate_model(nn, X_test, y_test)} %")
