import numpy as np
from sklearn import neighbors, metrics

raw_data = np.genfromtxt('plrx.txt')			#reading data from txt file

data = raw_data[:,:12]					#splicing the raw data
label = raw_data[:,-1]

print data.shape, label.shape				#shapes of the matrices

clf1 = neighbors.KNeighborsClassifier(n_neighbors= 2)	#nearest two neighbors

X = data						#data for training
y = label

clf1.fit(X, y)						#fitting the data with the model
result = clf1.predict(X)					#prediction with training data

training_accuracy = metrics.accuracy_score(y, result)	#training accuracy
print "The training accuracy is",100*training_accuracy,'%'

c_Matrix = metrics.confusion_matrix(y, result)		#confusion matrix
print "Confusion matrix:\n",c_Matrix
