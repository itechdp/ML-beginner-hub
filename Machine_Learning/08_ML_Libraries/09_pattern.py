import numpy as np
arr = np.ones((5,6))
col = 0
for ele_row in range(5):
    for ele in range(6):
        arr[ele_row,ele:] = col
        col+=1
print(arr)
'''
[[ 0.  1.  2.  3.  4.  5.] 
 [ 0.  6.  7.  8.  9. 10.] 
 [ 0. 11. 12. 13. 14. 15.] 
 [ 0. 16. 17. 18. 19. 20.] 
 [ 0. 21. 22. 23. 24. 25.]]
'''
        