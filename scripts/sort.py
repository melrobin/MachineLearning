import numpy as np
arr = input( 'What is the argument that you want sorted? ')

def sort(arr):
	if len(arr) <= 1:
		return arr
	pivot = arr[len(arr)/2]
	left = [x for x in arr if x < pivot]
	middle = [x for x in arr if x == pivot]
	right = [x for x in arr if x > pivot]
	return sort(left) + middle + sort(right)

print sort(arr)
print type(arr)
