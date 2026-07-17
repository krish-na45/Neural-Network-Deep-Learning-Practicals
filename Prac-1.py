import numpy as np

# Vector Addition
vector1 = np.array([1, 3, 5, 6])
vector2 = np.array([4, 6, 8, 7])

print("vector1:", vector1)
print("vector2:", vector2)

result = vector1 + vector2
print("Vector Addition Result:", result)

# Matrix Multiplication
matrix1 = np.array([[2, 5, 3], [4, 0, 2]])
matrix2 = np.array([[5, 2], [1, 4], [3, 7]])

print("matrix1:\n", matrix1)
print("matrix2:\n", matrix2)

result = np.matmul(matrix1, matrix2)
print("Matrix Multiplication Result:\n", result)
