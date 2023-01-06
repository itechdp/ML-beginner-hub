# Vector operations
import numpy as np 
vector1 = np.array([1,-1,2])
vector2 = np.array([2,5,2])
add = vector1+vector2 # Adding two vectors
sub = vector1-vector2 # Subtracting two vectors
mul = vector1*vector2 # Multiplicating two vectors
div = vector1/vector2 # Dividing two vectors
lengthv = np.linalg.norm(vector1) #it is used to get the multiplication of the array Result would be: 2.449489742783178 equivalent to Root 6
result_vector = np.dot(vector1,vector2)
print(result_vector) # Dot product 
print(add,sub,mul,div)
print(lengthv)

#Matrix Operations

matrix1 = np.array([[1,1,1],[2,2,2]]) # 2 X 3 Matrix 2 Rows and 3 columns  Rows--> [] [Columns--> 1,2,3]
matrix2 = np.array([[2,2,2],[3,3,3]])
print(matrix1)
print(matrix2)

resultant_add = np.add(matrix1,matrix2) #Addition of two matrix
print(resultant_add)

resultant_sub = np.subtract(matrix1,matrix2) #Subtraction of the matrix
print(resultant_sub)

dot_matrix1 = np.array([[1,2],[3,4]]) #Dot product of the matrix
dot_matrix2 = np.array([[5,6],[7,8]])
resultant_dotPR = np.dot(dot_matrix1,dot_matrix2)
print(resultant_dotPR)

det_matrix = np.array([[1,2],[3,4]]) # Determinant of the matrix
resultant_det = np.linalg.det(det_matrix)
print(resultant_det)

resultant_inv = np.linalg.inv(det_matrix) #Inverse of matrix
print(resultant_inv)
