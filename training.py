import numpy as np
from sklearn import neighbors, metrics, svm

raw_data = np.genfromtxt('plrx.txt')	#plrx.txt is the dataset txt file 

data = raw_data[:,:12]			#slicing the raw data
label = raw_data[:,-1]
X = data 				#data for testing
y = label 

def knn(X,y):								#k-NN classifier as a function
	clf1 = neighbors.KNeighborsClassifier( n_neighbors = 2)		#returns the predicted label values
	clf1.fit(X,y)
	results = clf1.predict(X)
	return results;

def vectors(X, y):							#SVM classifier as a function
	clf2 = svm.SVC(gamma = "auto")					#returns the predicted label values
	clf2.fit(X,y)
	results = clf2.predict(X)
	return results;

print "Using k-NN, the accuracy and confusion matrix is:", 100*metrics.accuracy_score(y, knn(X,y)),"\n",metrics.confusion_matrix(y, knn(X,y))
print "Using SVMs, the accuracy and confusion matrix is:", 100*metrics.accuracy_score(y, vectors(X,y)),"\n",metrics.confusion_matrix(y, vectors(X,y))


