import torch

# 1: Build a tensor from the original Python list and print shape and dtype
task3_1 = torch.tensor([[2, 3, 1], [5, -2, 1]])
print("Tensor 1:", task3_1)
print("Shape:", task3_1.shape)
print("Dtype:", task3_1.dtype)

# 2: Build a 3x4x2 tensor filled with random float numbers on [0,1] and print shape and values
"""
The first dimension (3): Represents 3 matrices (or 3 planes of 4×2 matrices).
The second dimension (4): Indicates that each matrix has 4 rows.
The third dimension (2): Shows that each row has 2 elements (columns).
"""
task3_2 = torch.rand(3, 4, 2)
print("\nTensor 2:", task3_2)
print("Shape:", task3_2.shape)

# 3: Build a 2x1x5 tensor filled with 1 and print shape and values
task3_3 = torch.ones(2, 1, 5)
print("\nTensor 3:", task3_3)
print("Shape:", task3_3.shape)

# 4: Matrix multiplication of two tensors
# 矩陣乘法 (Matrix Multiplication, Dot Product)
# 遵循 內積 (dot product) 計算規則
task3_3_4_a = torch.tensor([[1, 2, 4], [2, 1, 3]])
task3_3_4_b = torch.tensor([[5], [2], [1]])
result4 = torch.matmul(task3_3_4_a, task3_3_4_b)
print("\nMatrix multiplication result:")
print(result4)

# 5: Element-wise product of two tensors
# 元素逐項相乘 (Element-wise Product, Hadamard Product)
task3_5_a = torch.tensor([[1, 2], [2, 3], [-1, 3]])
task3_5_b = torch.tensor([[5, 4], [2, 1], [1, -5]])
result5 = task3_5_a * task3_5_b
print("\nElement-wise multiplication result:")
print(result5)
