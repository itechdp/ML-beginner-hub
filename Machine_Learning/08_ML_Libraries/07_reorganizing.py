# Reorganizing Array
import numpy as np
prev = np.array([[1,2,3,4,5],[6,7,8,9,10]])
after = prev.reshape((5,2))
print(after)

# Vertically stacking vecotors

vc1 = np.array([1,2,3,4,5])
vc2 = np.array([6,7,8,9,10])

stackv = np.vstack([vc1,vc2])
print(stackv)

# Horizontally stacking vecotrs
vc1 = np.array([1,2,3,4,5])
vc2 = np.array([6,7,8,9,10])
stackv = np.hstack([vc1,vc2])
print(stackv)

