import math
from abc import ABC, abstractmethod
from dataclasses import dataclass


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

    @abstractmethod
    def backward(self, output_losses):
        pass


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


loss_func = LossFunctionHelper()


print("------ Regression Tasks -------")
print("------------- Task 1 -------------")
nn = Regression(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, -0.5]], bias_weight=[0.6]),
        Layer(weight=[[0.6], [-0.3]], bias_weight=[0.4, 0.75]),
    ],
)
learning_rate = 0.01
outputs = nn.forward([1.5, 0.5])
expect_values = [0.8, 1]
total_losses = loss_func.calculate_mse(expect_values, outputs)
print(f"Total Loss", total_losses)
output_gradient = LossFunctionHelper.compute_derivative(expect_values, outputs, "mse")
nn.backward(output_gradient)
nn.zero_grad(learning_rate)
for i in range(len(nn.layers)):
    print("Layer", i)
    print("neuron_weight", nn.layers[i].weight)
    print("bias_weight", nn.layers[i].bias_weight)

print("------------- Task 2 -------------")

nn = Regression(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, -0.5]], bias_weight=[0.6]),
        Layer(weight=[[0.6], [-0.3]], bias_weight=[0.4, 0.75]),
    ],
)
learning_rate = 0.01
expect_values = [0.8, 1]
for i in range(1001):
    outputs = nn.forward([1.5, 0.5])
    total_losses = loss_func.calculate_mse(expect_values, outputs)
    loss_gradient = LossFunctionHelper.compute_derivative(expect_values, outputs, "mse")
    nn.backward(loss_gradient)
    nn.zero_grad(learning_rate)
print("Total_losses", total_losses)

print("================================================================", end="\n\n")
print("------ Binary Classification Tasks -------")

print("------------- Task 1 -------------")
nn = BinaryClassification(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, 0.4]], bias_weight=[-0.5]),
    ],
)
expect_values = [1.0]
learning_rate = 0.1

outputs = nn.forward([0.75, 1.25])
total_losses = loss_func.binary_cross_entropy(expect_values, outputs)
print(f"Total Loss", total_losses)

output_gradient = LossFunctionHelper.compute_derivative(expect_values, outputs, "bce")
nn.backward(output_gradient)
nn.zero_grad(learning_rate)
for i in range(len(nn.layers)):
    print("Layer", i)
    print("neuron_weight", nn.layers[i].weight)
    print("bias_weight", nn.layers[i].bias_weight)

print("------------- Task 2 -------------")

nn = BinaryClassification(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, 0.4]], bias_weight=[-0.5]),
    ],
)
expect_values = [1.0]
learning_rate = 0.1

for i in range(1001):
    outputs = nn.forward([0.75, 1.25])
    total_losses = loss_func.binary_cross_entropy(expect_values, outputs)
    loss_gradient = LossFunctionHelper.compute_derivative(expect_values, outputs, "bce")
    nn.backward(loss_gradient)
    nn.zero_grad(learning_rate)
print("Total_losses", total_losses)
