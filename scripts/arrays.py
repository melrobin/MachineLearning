
import numpy as np

def value(x,y):
	return x*y

array1 = np.array([1, 2, 3])
array2 = np.array([3, 2, 1])

print value(array1, array2) 
print array1.dot(array2)

