import numpy as np

#Inializing the Array
a = np.array([1,2,3,4,5],dtype='int16')
print(a)

dim = a.ndim
print(dim) # Getting the dimension of array

shape = a.shape
print(shape) # Getting the shape of array (Row,Column)

item = a.itemsize
print(item) # Getting the byte value of each element

byte = a.nbytes
print(byte) # Getting the total bytes taken in memory

typed = a.dtype
print((typed)) # Get the datatype of the array

