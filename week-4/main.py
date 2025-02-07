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


class LossFunctionHelper:
    def calculate_mse(self, expected_values, output_values):
        n = len(expected_values)
        squared_errors = 0

        for i in range(n):
            squared_errors += (expected_values[i] - output_values[i]) ** 2

        mse = squared_errors / n
        return mse

    def binary_cross_entropy(self, expected_values, output_values):
        n = len(expected_values)
        loss = 0
        for i in range(n):
            loss += expected_values[i] * math.log(output_values[i]) + (
                1 - expected_values[i]
            ) * math.log(1 - output_values[i])
        return -loss

    def categorical_cross_entropy(self, expected_values, output_values):
        epsilon = 1e-15
        loss = 0.0

        for e, p in zip(expected_values, output_values):
            # Clip predicted probability to avoid log(0)
            p = max(min(p, 1 - epsilon), epsilon)
            # Add to loss if expected value is non-zero
            loss += e * -math.log(p)

        return loss


class Network(ABC):
    def __init__(self, bias: float, layers: list[Layer]):
        self.bias = bias
        self.layers = layers
        self.output_data = None

    @abstractmethod
    def activation_function(self, input: float, layer_num: int):
        pass

    def forward(self, input_data: list):
        current_input = input_data
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

        return current_input


class Regression(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return ActivationFunctions.linear(input)


class BinaryClassification(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return ActivationFunctions.sigmoid(input)


class MultiLabelClassification(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return ActivationFunctions.sigmoid(input)


class MultiClassClassification(Network):
    def __init__(self, bias: float, layers: list[Layer]):
        super().__init__(bias, layers)

    def forward(self, input_data: list):
        current_input = input_data
        for i in range(len(self.layers)):
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
            if i == 1:
                layer_output = self.activation_function_all_inputs(layer_output)
            current_input = layer_output

        return current_input

    def activation_function_all_inputs(self, inputs):
        return ActivationFunctions.softmax(inputs)

    def activation_function(self, input: float, layer_num: int):
        if layer_num == 0:
            return ActivationFunctions.relu(input)
        return input


loss_func = LossFunctionHelper()
print("------ Regression Tasks -------")
nn = Regression(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, -0.5], [0.4, 0.5]], bias_weight=[0.6, -0.25]),
    ],
)

outputs = nn.forward([1.5, 0.5])
print(outputs)
expect_values = [0.8, 1]
print(f"Total Loss", loss_func.calculate_mse(expect_values, outputs))

outputs = nn.forward([0, 1])
print(outputs)
expect_values = [0.5, 0.5]
print(f"Total Loss", loss_func.calculate_mse(expect_values, outputs))

print("------ Binary Classification Tasks -------")


nn = BinaryClassification(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, 0.4]], bias_weight=[-0.5]),
    ],
)


outputs = nn.forward([0.75, 1.25])
print(outputs)
expect_values = [1]
print(f"Total Loss", loss_func.binary_cross_entropy(expect_values, outputs))

outputs = nn.forward([-1, 0.5])
print(outputs)
expect_values = [0]
print(f"Total Loss", loss_func.binary_cross_entropy(expect_values, outputs))

print("------ Multi-Label Classification Tasks -------")
nn = MultiLabelClassification(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(
            weight=[[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]], bias_weight=[0.6, 0.5, -0.5]
        ),
    ],
)

outputs = nn.forward([1.5, 0.5])
print(outputs)
expect_values = [1, 0, 1]
print(f"Total Loss", loss_func.binary_cross_entropy(expect_values, outputs))

outputs = nn.forward([0, 1])
print(outputs)
expect_values = [1, 1, 0]
print(f"Total Loss", loss_func.binary_cross_entropy(expect_values, outputs))


print("------ Multi-Class Classification Tasks -------")

nn = MultiClassClassification(
    bias=1,
    layers=[
        Layer(weight=[[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(
            weight=[[0.8, -0.4], [0.5, 0.4], [0.3, 0.75]], bias_weight=[0.6, 0.5, -0.5]
        ),
    ],
)
outputs = nn.forward([1.5, 0.5])
print(outputs)
expect_values = [1, 0, 0]
print(f"Total Loss", loss_func.categorical_cross_entropy(expect_values, outputs))

outputs = nn.forward([0, 1])
print(outputs)
expect_values = [0, 0, 1]
print(f"Total Loss", loss_func.categorical_cross_entropy(expect_values, outputs))
