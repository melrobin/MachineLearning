import numpy as np
from sklearn import svm, metrics

raw_data = np.genfromtxt('plrx.txt')		#reading data from txt file

data = raw_data[:,:12]				#slicing the raw data
label = raw_data[:,-1]

print data.shape, label.shape			#shape of the matrices

classifier = svm.SVC(kernel = 'rbf' )		#defining the classifier. Kernel= 'rbf', gamma= 0.083

X = data					#data for training
y = label

classifier.fit(X, y)				#fitting data with model

results = classifier.predict(X)			#prediction of training data

train_accuracy = metrics.accuracy_score(y, results)
print "The training accuracy is", 100*train_accuracy,'%'

c_Matrix = metrics.confusion_matrix(y, results)	#confusion matrix
print "Confusion matrix:\n", c_Matrix
