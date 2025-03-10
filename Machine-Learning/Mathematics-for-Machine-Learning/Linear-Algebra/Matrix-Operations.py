import numpy as np

class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        if data is not None:
            self.data = np.array(data, dtype=float)
        else:
            self.data = np.zeros((rows, cols), dtype=float)
    
    def populate(self, data):
        self.data = np.array(data, dtype=float)
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions to add")
        result = Matrix(self.rows, self.cols)
        result.data = self.data + other.data
        return result
    
    def __mul__(self, other):
        result = Matrix(self.rows, self.cols)
        if isinstance(other, (int, float)):  # Scalar multiplication
            result.data = self.data * other
            return result
        if self.cols != other.rows:
            raise ValueError("Matrices must have compatible dimensions to multiply")
        result = Matrix(self.rows, other.cols)
        result.data = np.matmul(self.data, other.data)
        return result

    @staticmethod
    def determinant(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate determinant")
        return np.linalg.det(matrix.data)

    @staticmethod
    def inverse(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate inverse")
        det = Matrix.determinant(matrix)
        if np.isclose(det, 0):
            raise ValueError("Matrix is singular and cannot be inverted")
        
        result = Matrix(matrix.rows, matrix.cols)
        result.data = np.linalg.inv(matrix.data)
        return result

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        result.data = np.transpose(self.data)
        return result

    @staticmethod
    def matminors(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate minors")
        
        n = matrix.rows
        minors = Matrix(n, n)
        for i in range(n):
            for j in range(n):
                # Create a matrix with row i and column j removed
                minor = np.delete(np.delete(matrix.data, i, axis=0), j, axis=1)
                minors.data[i, j] = np.linalg.det(minor)
        return minors

    @staticmethod
    def matcofactors(matrix):
        cofactors = Matrix(matrix.rows, matrix.cols)
        cofactors.data = matrix.data.copy()
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                cofactors.data[i, j] *= (-1) ** (i + j)
        return cofactors
    
    @staticmethod
    def row_echelon(matrix, inplace=False):
        if not inplace:
            result = Matrix(matrix.rows, matrix.cols, matrix.data.copy())
        else:
            result = matrix

        r = 0  # Row tracker
        for i in range(result.cols):
            # Find pivot row (starting from r)
            pivot_row = None
            for j in range(r, result.rows):
                if result.data[j, i] != 0:
                    pivot_row = j
                    break

            if pivot_row is None:
                continue  # No pivot found in this column

            # Swap pivot row with current row
            result = Matrix.row_switch(result, pivot_row, r, inplace)

            # Normalize pivot row
            pivot_value = result.data[r, i]
            if pivot_value != 0:
                result = Matrix.row_multiplication(result, r, 1 / pivot_value, inplace)

            # Eliminate entries below pivot
            for j in range(r + 1, result.rows):
                factor = result.data[j, i]
                result.data[j, :] -= factor * result.data[r, :]

            r += 1  # Move to the next row
            if r >= result.rows:
                break

        return result
    
    @staticmethod
    def row_multiplication(matrix, i, scalar, inplace=False):
        if inplace:
            matrix.data[i, :] *= scalar
            return matrix
        else:
            result = Matrix(matrix.rows, matrix.cols)
            result.data = matrix.data.copy()
            result.data[i, :] *= scalar
            return result

    @staticmethod
    def row_switch(matrix, i, j, inplace=False):
        if inplace:
            matrix.data[[i, j], :] = matrix.data[[j, i], :]
            return matrix
        else:
            result = Matrix(matrix.rows, matrix.cols)
            result.data = matrix.data.copy()
            result.data[[i, j], :] = matrix.data[[j, i], :]
            return result
    
    @staticmethod
    def row_addition(matrix, i, j, inplace=False):
        if inplace:
            matrix.data[i, :] += matrix.data[j, :]
            return matrix
        else:
            result = Matrix(matrix.rows, matrix.cols)
            result.data = matrix.data.copy()
            result.data[i, :] += matrix.data[j, :]
            return result


# Test code
Matrix1 = Matrix(2, 2, [[1, 2], [3, 4]])
Matrix2 = Matrix(2, 2, [[5, 6], [7, 8]])
Matrix3 = Matrix(2, 2, [[9, 10], [11, 12]])
Matrix4 = Matrix(3, 3, [[1, 2, 3], [4, 5, 6], [7, 15, 9]])
Matrix5 = Matrix(5, 5, [[0, 6, -2, -1, 5], [0, 0, 0, -9, -7], [0, 15, 35, 0, 0], [0, -1, -11, -2, 1], [-2, -2, 3, 0, -2]])

print("Determinant of Matrix1:", Matrix.determinant(Matrix1))
print("Determinant of Matrix2:", Matrix.determinant(Matrix2))
print("Determinant of Matrix4:", Matrix.determinant(Matrix4))
print("Determinant of Matrix5:", Matrix.determinant(Matrix5))
try:
    print("Inverse of Matrix4:")
    inverse_matrix4 = Matrix.inverse(Matrix4)
    for row in inverse_matrix4.data:
        print(row)
except ValueError as e:
    print("Error:", e)

try:
    print("Inverse of Matrix5:")
    inverse_matrix5 = Matrix.inverse(Matrix5)
    for row in inverse_matrix5.data:
        print(row)
except ValueError as e:
    print("Error:", e)

print("Row echelon form of Matrix1:")
for row in Matrix.row_echelon(Matrix1).data:
    print(row)
print("Row echelon form of Matrix2:")
for row in Matrix.row_echelon(Matrix2).data:
    print(row)
print("Row echelon form of Matrix4:")
for row in Matrix.row_echelon(Matrix4).data:
    print(row)
print("Row echelon form of Matrix5:")
for row in Matrix.row_echelon(Matrix5).data:
    print(row)