import numpy as np
file = np.genfromtxt('E:\\Programming languages\\Python\\Machine_Learning\\08_ML_Libraries\\data.txt',delimiter=',',dtype='int32')
print(file)

# Boolean and advance indexing
grtfivebool = file > 5 # Getting true false based on condition, condition applies on each element
grtfive = file[file>5] # Getting all the element of which the element is greater than five 
print(grtfive,grtfivebool)

arr = np.array([1,2,3,4,5])
print(arr[[1,4]]) # Accessing the multiple element at the same time

condData = ((file > 2) & (file < 10))
print(condData)