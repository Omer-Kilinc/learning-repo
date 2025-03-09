class Matrix:
    def __init__(self, rows, cols, data=None):
        self.rows = rows
        self.cols = cols
        self.data = [[0] * cols for _ in range(rows)]
        if data is not None:
            self.populate(data)
    
    def populate(self, data):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = data[i][j]
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must have the same dimensions to add")
        result = Matrix(self.rows, self.cols)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __mul__(self, other):
        if isinstance(other, (int, float)):  # Scalar multiplication
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result
        if self.cols != other.rows:
            raise ValueError("Matrices must have compatible dimensions to multiply")
        result = Matrix(self.rows, other.cols)
        for i in range(result.rows):
            for j in range(result.cols):
                for k in range(self.cols):
                    result.data[i][j] += self.data[i][k] * other.data[k][j]
        return result

    @staticmethod
    def determinant(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate determinant")

        if matrix.rows == 1:
            return matrix.data[0][0]

        if matrix.rows == 2:
            return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]

        det = 0
        for j in range(matrix.cols):
            submatrix = Matrix(matrix.rows - 1, matrix.cols - 1)
            sub_row = 0
            for row in range(1, matrix.rows):
                sub_col = 0
                for col in range(matrix.cols):
                    if col == j:
                        continue
                    submatrix.data[sub_row][sub_col] = matrix.data[row][col]
                    sub_col += 1
                sub_row += 1
            det += matrix.data[0][j] * Matrix.determinant(submatrix) * ((-1) ** j)
        return det

    @staticmethod
    def inverse(matrix):
        det = Matrix.determinant(matrix)
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        minors = Matrix.matminors(matrix)
        cofactors = Matrix.matcofactors(minors)
        adjugate = cofactors.transpose()
        return adjugate * (1 / det)

    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result

    @staticmethod
    def matminors(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate minors")

        minors = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                submatrix = Matrix(matrix.rows - 1, matrix.cols - 1)
                sub_row = 0
                for row in range(matrix.rows):
                    if row == i:
                        continue
                    sub_col = 0
                    for col in range(matrix.cols):
                        if col == j:
                            continue
                        submatrix.data[sub_row][sub_col] = matrix.data[row][col]
                        sub_col += 1
                    sub_row += 1
                minors.data[i][j] = Matrix.determinant(submatrix)
        return minors

    @staticmethod
    def matcofactors(matrix):
        cofactors = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                cofactors.data[i][j] = matrix.data[i][j] * ((-1) ** (i + j))
        return cofactors


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
