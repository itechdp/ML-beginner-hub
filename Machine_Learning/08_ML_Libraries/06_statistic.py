#Statestics operation
import numpy as np
stats = np.array([[1,2,3],[4,5,6]])
print(stats)
mins = np.min(stats) # Used to get min of the array(array,axis)
print(mins)
maxs = np.max(stats) # Used to get max of the array(array,axis)
print(maxs)
sums = np.sum(stats) # Used to get sum of the array(array,axis)
print(sums)
medians = np.median(stats) #Used to get middle value of array
print(medians)
means = np.mean(stats) # Used to get average of array 
print(means)