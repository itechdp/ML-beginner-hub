# Accessing/Changing specific elements, rows, columns, etc.
import numpy as np
a = np.array([[1,2,3,4],[5,6,7,8]])
print(a)

ele = a[1,1]
print(ele) # Using to get specific element from the list using, array[row,column]

spe_row = a[0,:] 
print(spe_row) #Used to get row which index is 0 array[row index, ending point]

spe_col = a[:,2] 
print(spe_col) # Used to get specific column value array[ :, get index value of column]

get_block = a[0,1::2] 
print(get_block) # Used to get specific block of element array[rowindex,startindex:endindex:skipelement] 

a[0,3] = 5 # Changing the 0th row index's 3rd index value
a[1,0] = 4 # Changing the 1st row index's 0th index value
print(a) 

a[:,1] = [3,4] # Changing the value of column at index 1 in each row
print(a)

b = np.array([[[1,2,3]],[[4,5,6]]]) # Creating 3-D Array
print(b[0,0,2]) # Getting the specific element from the 3d Array(TotalMetcies,rowindex,colindex)

b[:,0,1] = 6
print(b)