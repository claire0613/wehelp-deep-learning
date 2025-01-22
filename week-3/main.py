from dataclasses import dataclass


@dataclass
class Layer:
    weight: list
    bias_weight: list


class Network:
    def __init__(
        self,
        bias: float,
        layers: list[Layer],
    ):
        self.bias = bias
        self.layers = layers
        self.output_data = None

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
                layer_output.append(element)
            current_input = layer_output

        return current_input


nn = Network(
    bias=1,
    layers=[
        Layer([[0.5, 0.2], [0.6, -0.6]], bias_weight=[0.3, 0.25]),
        Layer(weight=[[0.8, 0.4]], bias_weight=[-0.5]),
    ],
)
print("-------------------- Neural network 1 --------------------")
outputs = nn.forward([1.5, 0.5])
print(outputs)

outputs = nn.forward([0, 1])
print(outputs)


nn = Network(
    bias=1,
    layers=[
        Layer([[0.5, 1.5], [0.6, -0.8]], bias_weight=[0.3, 1.25]),
        Layer([[0.6, -0.8]], bias_weight=[0.3]),
        Layer(weight=[[0.5], [-0.4]], bias_weight=[0.2, 0.5]),
    ],
)
print("-------------------- Neural network 2 --------------------")

outputs = nn.forward([0.75, 1.25])
print(outputs)

outputs = nn.forward([-1, 0.5])
print(outputs)
