

class Matrix:
    def __init__(self, rows, cols, data = None):
        self.rows = rows
        self.cols = cols
        self.data = []
        for i in range(rows):
            self.data.append([])
            for j in range(cols):
                self.data[i].append(0)
        
        if data != None:
            self.populate(data)
    
    def populate(self, data):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = data[i][j]

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrices must be of same dimensions to add")
        
        result = Matrix(self.rows, self.cols)
        for i in range (self.rows):
            for j in range(self.cols):
                result.data[i][j] = self.data[i][j] + other.data[i][j]
        return result
    
    def __mul__(self, other):

        if type[other] == int or type[other] == float:
            result = Matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    result.data[i][j] = self.data[i][j] * other
            return result       

        if self.cols != other.rows:
            raise ValueError("Matrices must be of compatible dimensions to multiply")
        
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
            return Matrix.data[0][0]
        
        if matrix.rows == 2:
            return matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]
        
        det = 0
        
        for i in range(Matrix.columns):
            submatrix = Matrix(matrix.rows - 1, matrix.cols - 1)
            sub_row = 0
            for row in range(1, matrix.rows):
                sub_col = 0
                for col in range(matrix.cols):
                    if col != j:
                        submatrix.data[sub_row][sub_col] = matrix.data[row][col]
                        sub_col += 1
                sub_row += 1
            det += matrix.data[0][j] * Matrix.determinant(submatrix) * ((-1) ** i)
        
        return det
        
    @staticmethod
    def inverse(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to invert")
        
        result = Matrix(matrix.rows, matrix.cols)
        result = (1/matrix.determinant(matrix)) * matrix.transpose()
    
    def transpose(self):
        result = Matrix(self.cols, self.rows)
        for i in range(self.rows):
            for j in range(self.cols):
                result.data[j][i] = self.data[i][j]
        return result
    
    @staticmethod
    def matminors(matrix):
        if matrix.rows != matrix.cols:
            raise ValueError("Matrix must be square to calculate matrix of minors")
        
        # Create a new matrix to store the minors
        minors = Matrix(matrix.rows, matrix.cols)
        
        # For each element in the matrix
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                # Create submatrix by removing row i and column j
                submatrix = Matrix(matrix.rows - 1, matrix.cols - 1)
                
                # Fill the submatrix
                sub_row = 0
                for row in range(matrix.rows):
                    if row == i:
                        continue  # Skip the row we're removing
                        
                    sub_col = 0
                    for col in range(matrix.cols):
                        if col == j:
                            continue  # Skip the column we're removing
                            
                        submatrix.data[sub_row][sub_col] = matrix.data[row][col]
                        sub_col += 1
                        
                    sub_row += 1
                    
                # Calculate the determinant of this submatrix
                minors.data[i][j] = Matrix.determinant(submatrix)




        
        


Matrix1 = Matrix(2, 2, [[1, 2], [3, 4]])
Matrix2 = Matrix(2, 2, [[5, 6], [7, 8]])
Matrix3 = Matrix(2, 2, [[9, 10], [11, 12]])

print(Matrix.determinant(Matrix1), "\n")
print(Matrix.determinant(Matrix2), "\n")
print(Matrix.determinant(Matrix3), "\n")

