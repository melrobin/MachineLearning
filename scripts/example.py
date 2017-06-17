import numpy as np

a = np.array([1,4,5,2])
b = np.array([3,4,5,2])

def sort(x):
	if x >= 5:
		return "Blue Team"
	elif x >= 3:
		return "Green Team"
	elif x > 0:
		return "Violet Team"


print("For array 'a':")
for x,y in enumerate(a,1):
	print '#',x,sort(y),':',y

print("For array 'b':")
for x,y in enumerate(b,1):
	print x,sort(y), y
	
