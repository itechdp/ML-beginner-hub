import numpy as np
# All the 0s matrix
zero = np.zeros((2,3)) # array(shape)
print(zero)

# All the 1s matrix
one = np.ones((3,3,3),dtype='int16') #array(shape,datatype)
print(one)

# any number matrix 
anymat = np.full(3,23) # array(shape,number)
print(anymat)

# Get the any number array like other array's shape
anylike = np.full_like(zero,30)
print(anylike)

#Random decimal number array
deci = np.random.rand(2,2) #array(shape)
print(deci)

# Random int number array
intarr = np.random.randint(10,size=(2,3)) # ArraY(range of integer,size=(shape))
print(intarr)

#Used to print identity matrix of any number
inden = np.identity(2)
print(inden)

# Used to repeat the array
arr = np.array([1,2,3])
rep = np.repeat(arr,3,axis=0)
print(rep)

#Printing the matrix
'''
[[1 1 1 1 1]
 [1 0 0 0 1]
 [1 0 9 0 1]
 [1 0 0 0 1]
 [1 1 1 1 1]]
'''
outside = np.ones((5,5),dtype='int16')
inside = np.zeros((3,3),dtype='int16')
inside[1,1] = 9
outside[1:-1,1:-1] = inside
print(outside)

# used to copy array
arr2 = np.array([1,2,3])
b = arr2.copy()
print(b)