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
        if isinstance(other, (int, float)):  # Scalar multiplication
            result = Matrix(self.rows, self.cols)
            result.data = self.data * other
            return result
        if self.cols != other.rows:
            raise ValueError("Matrices must have compatible dimensions to multiply")
        result = Matrix(self.rows, other.cols)
        result.data = np.dot(self.data, other.data)
        return result

    @staticmethod
    def determinant(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate determinant")
        
        if matrix.rows == 1:
            return matrix.data[0, 0]
        elif matrix.rows == 2:
            return matrix.data[0, 0] * matrix.data[1, 1] - matrix.data[0, 1] * matrix.data[1, 0]
        else:
            det = 0
            for i in range(matrix.cols):
                minor = Matrix(matrix.rows - 1, matrix.cols - 1)
                minor.data = np.delete(np.delete(matrix.data, 0, axis=0), i, axis=1)
                det += ((-1) ** i) * matrix.data[0, i] * Matrix.determinant(minor)
            return det

    @staticmethod
    def inverse(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate inverse")
        
        det = Matrix.determinant(matrix)
        if np.isclose(det, 0):
            raise ValueError("Matrix is singular and cannot be inverted")
        
        cofactors = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                minor = Matrix(matrix.rows - 1, matrix.cols - 1)
                minor.data = np.delete(np.delete(matrix.data, i, axis=0), j, axis=1)
                cofactors.data[i, j] = ((-1) ** (i + j)) * Matrix.determinant(minor)
        
        adjugate = cofactors.transpose()
        inverse = Matrix(matrix.rows, matrix.cols)
        inverse.data = adjugate.data / det
        return inverse

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        result.data = np.transpose(self.data)
        return result

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
            result.data[[r, pivot_row], :] = result.data[[pivot_row, r], :]

            # Normalize pivot row
            pivot_value = result.data[r, i]
            if pivot_value != 0:
                result.data[r, :] /= pivot_value

            # Eliminate entries below pivot
            for j in range(r + 1, result.rows):
                factor = result.data[j, i]
                result.data[j, :] -= factor * result.data[r, :]

            r += 1  # Move to the next row
            if r >= result.rows:
                break

        return result
    
    @staticmethod
    def reduced_row_echelon(matrix, inplace=False):
        if not inplace:
            result = Matrix(matrix.rows, matrix.cols, matrix.data.copy())
        else:
            result = matrix

        # Perform forward elimination to get REF
        result = Matrix.row_echelon(result, inplace=True)

        # Iterate through the rows in reverse order for back-substitution
        for i in range(result.rows - 1, -1, -1):
            row = result.data[i]
            
            # Find the pivot column (first nonzero entry in the row)
            pivot_col = next((j for j in range(result.cols) if row[j] != 0), None)
            if pivot_col is None:
                continue  # Skip all-zero rows
            
            # Normalize the pivot to 1
            pivot_value = row[pivot_col]
            if pivot_value != 0:  # Avoid division by zero
                row /= pivot_value

            # Eliminate entries above the pivot
            for k in range(i):  # Iterate over rows above the current row
                factor = result.data[k, pivot_col]
                result.data[k] -= factor * row

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
    
    @staticmethod
    def Moore_penrose(matrix, inplace=False):
        if inplace:
            matrix.data = Matrix.inverse(matrix.data * Matrix.transpose(matrix.data))
            return matrix
        else:
            result = Matrix(matrix.rows, matrix.cols)
            result.data = Matrix.inverse(matrix.data * Matrix.transpose(matrix.data))
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